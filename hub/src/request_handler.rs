use std::time::Duration;

use anyhow::bail;
use log::{error, info};

use tokio_graceful_shutdown::{SubsystemBuilder, Toplevel};
use tonic::{transport::Server, Code, Request, Response, Status};
use tonic_types::{ErrorDetails, StatusExt};

use grpc::sniper::sniper_hub_server::{SniperHub, SniperHubServer};
use grpc::sniper::{
    start_sample_option, DataType, HelloRequest, HelloResponse, StartSampleOption, TensorMessage,
    VoidMessage,
};

use grpc::sniper::{TensorProto, TensorShapeProto};
use util::send_error_response;

use super::pipeline::GroupSamplePipeline;
use super::pipeline::SingleSamplePipeline;

use super::sample::SampleBatch;

/// Hub server.
///
/// Start hub pipelien for processing input data, response to grpc requests from trainer.
#[derive(Debug)]
pub struct Hub {
    /// Sender, pass to pipeline.
    sample_batch_sender: async_channel::Sender<SampleBatch>,

    /// Receiver, for read sample.
    sample_batch_receiver: async_channel::Receiver<SampleBatch>,
}

impl Hub {
    pub fn new() -> Self {
        let (s, r) = async_channel::bounded::<SampleBatch>(100);

        Self {
            sample_batch_sender: s,
            sample_batch_receiver: r,
        }
    }
}

#[tonic::async_trait]
impl SniperHub for Hub {
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
        info!("start sample start");

        let mut response = VoidMessage::new();
        let mut err_details = ErrorDetails::new();

        // Check options.
        let request_inner = request.into_inner();

        info!("start sample request.seq_id: {}", request_inner.seq_id);

        if request_inner.options.is_none() {
            err_details.add_bad_request_violation("options", "options is None");
        }

        if err_details.has_bad_request_violations() {
            return send_error_response::<VoidMessage>(err_details);
        }

        let start_sample_option_res = request_inner.options.unwrap().to_msg::<StartSampleOption>();

        // Check StartSampleOption.
        if start_sample_option_res.is_err() {
            err_details
                .add_bad_request_violation("options", "options is invalid StartSampleOption");
        }

        if err_details.has_bad_request_violations() {
            return send_error_response::<VoidMessage>(err_details);
        }

        let mut start_sample_option = start_sample_option_res.unwrap();

        start_sample_option.parallel = num_cpus::get() as i32;

        let new_sender = self.sample_batch_sender.clone();

        // Start running.
        if start_sample_option.need_batch {
            // Need to rm unwrap. use other ways.
            let mut single_sample_pipeline =
                SingleSamplePipeline::new(start_sample_option.clone(), new_sender);

            if !single_sample_pipeline.init() {
                error!("single_sample_pipeline init failed!");
            }

            info!("before spawn pipeline!");

            tokio::spawn(async move {
                Toplevel::new(|s| async move {
                    s.start(SubsystemBuilder::new("single_sample_pipeline", |a| {
                        single_sample_pipeline.run(a)
                    }));
                })
                .catch_signals()
                .handle_shutdown_requests(Duration::from_millis(1000))
                .await;
            });
        } else {
            // Need to rm unwrap. use other ways.
            let mut group_sample_pipeline =
                GroupSamplePipeline::new(start_sample_option.clone(), new_sender);

            if !group_sample_pipeline.init() {
                error!("group_sample_pipeline init failed!");
            }

            tokio::spawn(async move {
                Toplevel::new(|s| async move {
                    s.start(SubsystemBuilder::new("group_sample_pipeline", |a| {
                        group_sample_pipeline.run(a)
                    }));
                })
                .catch_signals()
                .handle_shutdown_requests(Duration::from_millis(1000))
                .await
            });
        }

        info!("start_sample handle done!");

        Ok(Response::new(response))
    }

    async fn read_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        // TODO
        let mut response = TensorMessage::new();

        Ok(Response::new(response))
    }

    async fn heartbeat(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::new();

        Ok(Response::new(response))
    }
}
