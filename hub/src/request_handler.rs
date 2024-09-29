use std::borrow::BorrowMut;
use std::time::Duration;

use anyhow::bail;
use log::{error, info};
use std::cell::RefCell;
use sync_unsafe_cell::SyncUnsafeCell;
use util::histogram::{Histogram, HistogramAggregator, HistogramDetail, HistogramType};

use std::collections::HashMap as StdHashMap;
use std::sync::Arc;

use tokio::sync::mpsc;

use prost_types::Any;
use tokio_graceful_shutdown::{SubsystemBuilder, Toplevel};
use tonic::{transport::Server, Code, Request, Response, Status};
use tonic_types::{ErrorDetails, StatusExt};

use grpc::sniper::sniper_server::{Sniper, SniperServer};
use grpc::sniper::{
    start_sample_option, DataType, HelloRequest, HelloResponse, ReadSampleOption, Role,
    StartSampleOption, TensorMessage, VoidMessage,
};

use grpc::sniper::{TensorProto, TensorShapeProto};
use grpc::tool::{get_request_inner_options, send_bad_request_error, send_error_message};

use super::pipeline::GroupSamplePipeline;
use super::pipeline::SingleSamplePipeline;

use super::sample::SampleBatch;

/// Hub server.
///
/// Start hub pipelien for processing input data, response to grpc requests from trainer.
pub struct Hub {
    /// Receiver, for read sample.
    ///
    /// Why use `Arc<SyncUnsafe<Option>>` and where is the sender ?
    ///
    /// The channel is closed only after all sender are dropped.
    ///
    /// If we make sender and receiver both members of `Hub`, we need to construct them
    /// in `Hub::new`. The sender will be cloned to different thread. And because the `Hub`
    /// is always `&self`, there will always be an extra `self.sender` exists. There are no
    /// place to drop it, because drop the last sender would need `Hub` to be `&mut self`.
    /// It's impossible in `tonic`.
    ///
    /// Because the receiver is not cloned, so we can construct the sender and receiver in
    /// `self.start_sample`, and save the receiver to `self.sample_batch_receiver`. Then
    /// all cloned senders are moved into the spawned task, the origin sender would be dropped
    /// manually at the end of `self.start_sample` function. After spawned tasks finished, all
    /// senders will be dropped, and the receiver would be closed normally.
    sample_batch_receiver: Arc<SyncUnsafeCell<Option<async_channel::Receiver<SampleBatch>>>>,
}

impl Hub {
    pub fn new() -> Self {
        Self {
            sample_batch_receiver: Arc::new(SyncUnsafeCell::new(None)),
        }
    }

    fn get_histogram_aggregator(
        histogram_receiver: mpsc::Receiver<HistogramDetail>,
    ) -> HistogramAggregator {
        let histogram_types = vec![
            HistogramType::HubStartSample,
            HistogramType::HubReadSample,
            HistogramType::HubNext,
            HistogramType::HubProcess,
            HistogramType::HubFeedSample,
            HistogramType::HubReadMessage,
            HistogramType::HubCountMessage,
            HistogramType::HubBatchAssembler,
            HistogramType::HubCountAfterItemFilter,
            HistogramType::HubCountAfterLabelExtractor,
            HistogramType::HubCountSamplePos,
            HistogramType::HubCountSampleNeg,
            HistogramType::HubDecompress,
            HistogramType::HubParseProto,
        ];

        HistogramAggregator::new(histogram_receiver, &histogram_types)
    }
}

#[tonic::async_trait]
impl Sniper for Hub {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloResponse>, Status> {
        info!("say hello");

        let response = HelloResponse {
            message: format!("Hello {}!", request.into_inner().name).into(),
        };

        Ok(Response::new(response))
    }

    /// Start the SingleSamplePipeline or GroupSamplePipeline based on the parameter.
    async fn start_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let mut response = VoidMessage::default();

        let request_inner = request.into_inner();
        let mut start_sample_option =
            match get_request_inner_options::<StartSampleOption>(&request_inner) {
                Some(x) => x,
                None => {
                    return send_bad_request_error(
                        "options",
                        "options is invalid StartSampleOption",
                    );
                }
            };
        start_sample_option.parallel = num_cpus::get() as i32;

        let (batch_sender, batch_receiver) = async_channel::bounded::<SampleBatch>(100);

        unsafe {
            (*self.sample_batch_receiver.get()).insert(batch_receiver);
        }

        let (histogram_sender, histogram_receiver) = mpsc::channel::<HistogramDetail>(100);

        let histogram = Histogram::new(histogram_sender.clone());

        let histogram_aggregator = Self::get_histogram_aggregator(histogram_receiver);

        // Start running.
        if start_sample_option.need_batch {
            let mut single_sample_pipeline =
                SingleSamplePipeline::new(start_sample_option.clone(), batch_sender, histogram);

            if !single_sample_pipeline.init().await {
                error!("single_sample_pipeline init failed!");
            }

            tokio::spawn(async move {
                Toplevel::new(|s| async move {
                    s.start(SubsystemBuilder::new("single_sample_pipeline", |a| {
                        single_sample_pipeline.run(a)
                    }));
                    s.start(SubsystemBuilder::new("hub_histogram_aggregator", |a| {
                        histogram_aggregator.run(a)
                    }));
                })
                .catch_signals()
                .handle_shutdown_requests(Duration::from_millis(1000))
                .await;
            });
        } else {
            // Need to rm unwrap. use other ways.
            let mut group_sample_pipeline =
                GroupSamplePipeline::new(start_sample_option.clone(), batch_sender, histogram);

            if !group_sample_pipeline.init().await {
                error!("group_sample_pipeline init failed!");
            }

            tokio::spawn(async move {
                Toplevel::new(|s| async move {
                    s.start(SubsystemBuilder::new("group_sample_pipeline", |a| {
                        group_sample_pipeline.run(a)
                    }));
                    s.start(SubsystemBuilder::new("hub_histogram_aggregator", |a| {
                        histogram_aggregator.run(a)
                    }));
                })
                .catch_signals()
                .handle_shutdown_requests(Duration::from_millis(1000))
                .await
            });
        }

        Ok(Response::new(response))
    }

    /// Read one batch from sample_batch_receiver channel.
    async fn read_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        let mut response = TensorMessage::default();
        let mut err_details = ErrorDetails::new();

        let mut response = TensorMessage::default();

        let mut read_sample_option = ReadSampleOption::default();

        {
            let receiver = unsafe { &*self.sample_batch_receiver.get() };

            match receiver.as_ref() {
                Some(receiver) => {
                    // If receiver is empty, need to wait for next batch.
                    if receiver.is_empty() {
                        read_sample_option.need_wait = true;
                    }

                    // If receiver is closed, then hub task is done.
                    if receiver.is_closed() {
                        read_sample_option.over = true;
                    }
                }
                None => {
                    return send_error_message::<TensorMessage>("sample batch receiver is None!");
                }
            }
        }

        // options in TensorMessage.
        let options = Any::from_msg(&read_sample_option);

        if options.is_err() {
            return send_bad_request_error::<TensorMessage>(
                "options",
                "from read_sample_option to options failed!",
            );
        }

        // If need wait or is over, return to trainer.
        if read_sample_option.need_wait || read_sample_option.over {
            response.options = Some(options.unwrap());
            return Ok(Response::new(response));
        }

        let mut receiver_mut = unsafe { &mut *self.sample_batch_receiver.get() };

        match receiver_mut.as_mut() {
            Some(receiver) => match receiver.recv().await {
                Ok(sample_batch) => {
                    let batch_id = sample_batch.batch_id;

                    read_sample_option.batch_id = sample_batch.batch_id;

                    // Save labels to tensor1 of TensorMessage.
                    let dim_label = vec![1 as i64, sample_batch.batch_size as i64];
                    let tensor1 = TensorProto::with_vec(
                        DataType::DtInt32.into(),
                        &dim_label,
                        &sample_batch.labels,
                    );

                    // Convert dense features to 1d vec first, and save to tensor2 of TensorMessage.
                    let dims = vec![
                        sample_batch.batch_size as i64,
                        sample_batch.dense_total_size as i64,
                    ];

                    let mut vec = Vec::with_capacity(sample_batch.dense_total_size);
                    for multi_dense in sample_batch.dense_features.iter() {
                        for dense_features in multi_dense.iter() {
                            vec.extend_from_slice(dense_features);
                        }
                    }

                    let tensor2 = TensorProto::with_vec(DataType::DtFloat.into(), &dims, &vec);

                    response.role = Role::Hub.into();
                    response.role_id = Into::<i32>::into(Role::Hub) as u32;
                    response.seq_id = sample_batch.batch_id;

                    let options = Any::from_msg(&read_sample_option);
                    response.options = Some(options.unwrap());

                    response.tensor1 = Some(tensor1);
                    response.tensor2 = Some(tensor2);

                    Ok(Response::new(response))
                }
                Err(err) => send_bad_request_error::<TensorMessage>(
                    "options",
                    "from read_sample_option to options failed!",
                ),
            },
            None => {
                return send_error_message::<TensorMessage>("sample_batch_receiver is None!");
            }
        }
    }

    /// Update ps shard when auto shard is in active.
    async fn update_hub_shard(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn heartbeat(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        Ok(Response::new(TensorMessage::default()))
    }

    // Below are services for ps, no need for implementation for hub.
    async fn create(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn freeze(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn feed_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn push(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn pull(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        Ok(Response::new(TensorMessage::default()))
    }

    async fn embedding_lookup(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        Ok(Response::new(TensorMessage::default()))
    }

    async fn push_grad(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn save(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }

    async fn restore(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        Ok(Response::new(TensorMessage::default()))
    }

    async fn complete(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        Ok(Response::new(VoidMessage::default()))
    }
}
