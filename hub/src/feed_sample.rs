use anyhow::{anyhow, bail, Result};
use grpc::sniper::{FeedFieldInfo, GpuPsFeature64, Role, StartSampleOption, TensorMessage};
use log::{error, info};

use hashbrown::HashMap;

use prost::Message;
use prost_types::Any;
use ps::get_ps_client;
use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};
use util::{error_bail, FeaturePlacement};

use super::sample::SampleBatch;
use grpc::sniper::sniper_ps_client::SniperPsClient;
use grpc::sniper::TensorProto;
use grpc::sniper::{sniper_hub_client::SniperHubClient, FeedSampleOption};

/// Send features to ps and trainer.
///
/// For each SampleBatch, first generate a random u64 as batch_id, then send batch_id and sparse
/// features to ps, and send batch_id, dense_features and labels to trainer.
///
/// For each sparse field, we need to decide which ps to send. It's determined by FeaturePlament.
pub struct FeedSample {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Receiver from BatchAssembler or BatchReader.
    sample_batch_receiver: async_channel::Receiver<SampleBatch>,

    /// FeaturePlacement is used to determine which ps each var live in.
    feature_placement: FeaturePlacement,

    /// Total batch.
    total_batch: i64,

    /// Send batch_id, dense features, labels to trainer.
    trainer_data_sender: async_channel::Sender<SampleBatch>,
}

impl FeedSample {
    pub fn new(
        option: StartSampleOption,
        sample_batch_receiver: async_channel::Receiver<SampleBatch>,
        feature_placement: FeaturePlacement,
        trainer_data_sender: async_channel::Sender<SampleBatch>,
    ) -> Self {
        Self {
            option,
            sample_batch_receiver,
            feature_placement,
            total_batch: 0,
            trainer_data_sender,
        }
    }

    /// Initialize.
    pub fn init(&mut self) -> bool {
        true
    }

    /// Send batch_id and sparse features to ps.
    pub async fn send_to_ps(
        &mut self,
        sample_batch: &SampleBatch,
        ps_client: &mut SniperPsClient<tonic::transport::Channel>,
    ) -> Result<()> {
        if self.option.feature_list.is_none() {
            error_bail!("option.feature_list is none!");
        }

        let feature_list = self.option.feature_list.as_ref().unwrap();

        // For extensability, the value of sparse_features is a HashMap of String
        // and FeedSampleOption. Each key of the inner map is ps_endpoint. Right now
        // there are only one ps_endpoint for each sparse embedding table.
        //
        // EmbeddingTable -> ps_endpoint -> FeedSamleOption
        let mut sparse_features: HashMap<String, HashMap<String, FeedSampleOption>> =
            HashMap::new();

        // Iterator all sparse feature signs, get the corresponding ps address.
        for (i, signs) in sample_batch.sparse_signs.iter().enumerate() {
            if i >= feature_list.sparse_emb_table.len() {
                error_bail!(
                    "out of range, i: {}, feature_list.sparse_emb_table.len(): {}",
                    i,
                    feature_list.sparse_emb_table.len()
                );
            }

            let varname = &feature_list.sparse_emb_table[i];

            // Get ps shard.
            let shard_endpoints_opt = self.feature_placement.get_emb_placement(varname);
            if shard_endpoints_opt.is_none() {
                error_bail!(
                    "cannot get shard_endpoints for varname: {}",
                    varname.clone()
                );
            }

            let shard_endpoints = shard_endpoints_opt.unwrap();
            if shard_endpoints.len() == 0 {
                error_bail!("shard_endpoints.len() is 0, varname: {}", varname.clone());
            }

            // Get all data send to ps.
            // ps_endpoint -> FeedSampleOption
            let mut one_features: HashMap<String, FeedSampleOption> = HashMap::new();

            // Assembly FeedSampleOption.
            let mut feed_sample_option = FeedSampleOption::default();
            feed_sample_option.batch_size = sample_batch.batch_size as u32;
            feed_sample_option.work_mode = self.option.work_mode;

            // Assembly field_info.
            let mut field_info = FeedFieldInfo::default();

            field_info.field_idx = i as i32;
            field_info.field_dim = feature_list.sparse_field_count[i] as i32;

            if i >= sample_batch.item_indexes.len() {
                error_bail!(
                    "out of range, i: {}, sample_batch.item_indexes.len(): {}",
                    i,
                    sample_batch.item_indexes.len(),
                );
            }

            // GpuPsFeature64
            let mut gpu_ps_feature64 = GpuPsFeature64::default();

            gpu_ps_feature64.features.extend_from_slice(&signs);

            let indexes: Vec<i32> = sample_batch.item_indexes[i]
                .iter()
                .map(|x| *x as i32)
                .collect();
            gpu_ps_feature64.item_indices.extend_from_slice(&indexes);

            let mut buf = Vec::new();
            match gpu_ps_feature64.encode(&mut buf) {
                Ok(_) => {}
                Err(err) => {
                    error_bail!("gpu_ps_feature64 encode to str failed! error: {}", err);
                }
            }

            field_info.feature = buf;

            feed_sample_option.field_info.push(field_info);

            one_features.insert(shard_endpoints[0].clone(), feed_sample_option);

            // insert varname and signs to sparse_features
            sparse_features.insert(varname.clone(), one_features);
        }

        // Send the signs to ps. Wait for the response too check whether it's success.
        let mut handles = Vec::new();

        for (varname, inner) in sparse_features.iter() {
            for (ps_endpoint, feed_sample_option) in inner.iter() {
                let new_varname = varname.clone();
                let options = Any::from_msg(feed_sample_option)?;

                let batch_id = sample_batch.batch_id;

                let mut client = ps_client.clone();

                handles.push(tokio::spawn(async move {
                    // TODO: use real seq_id,
                    let tensor_message = TensorMessage {
                        role: Role::Hub.into(),
                        role_id: Into::<i32>::into(Role::Hub) as u32,
                        seq_id: batch_id,
                        varname: new_varname,
                        options: Some(options),
                        tensor1: Some(TensorProto::default()),
                        tensor2: Some(TensorProto::default()),
                    };

                    let request = tonic::Request::new(tensor_message);

                    client.feed_sample(request).await
                }));
            }
        }

        for handle in handles {
            match handle.await {
                Ok(_) => {}
                Err(err) => {
                    error!("Hub feed sample failed! error: {}", err);
                }
            }
        }

        Ok(())
    }

    /// Send batch_id, dense features and labels to channel.
    pub async fn send_to_trainer(&mut self, sample_batch: SampleBatch) -> Result<()> {
        match self.trainer_data_sender.send(sample_batch).await {
            Ok(_) => Ok(()),
            Err(err) => {
                error_bail!("send data to channel for trainer failed! error: {}", err);
            }
        }
    }

    /// Process the sample batch, send feature to ps and trainer.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        // Each thread get a ps_client.
        let mut ps_client = match get_ps_client().await {
            Ok(ps) => ps,
            Err(err) => {
                error_bail!("get_ps_client in hub failed! ar: {}", err);
            }
        };

        loop {
            tokio::select! {
                sample_batch_res = self.sample_batch_receiver.recv() => {
                    match sample_batch_res {
                        Ok(sample_batch) => {
                            self.total_batch += 1;

                            info!(
                                "get one batch! total_batch: {}, batch_id: {}",
                                self.total_batch,
                                sample_batch.batch_id.clone(),
                            );

                            match self.send_to_ps(&sample_batch, &mut ps_client).await {
                                Ok(_) => {},
                                Err(err) => {
                                    error!("send_to_ps failed! err: {}", err);
                                }
                            }

                            match self.send_to_trainer(sample_batch).await {
                                Ok(_) => {},
                                Err(err) => {
                                    error!("send_to_trainer failed! err: {}", err);
                                }
                            }
                        },
                        Err(err) => {
                            error_bail!("get sample batch failed! error: {}", err);
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("FeedSample shutdown!");
                    return Ok(());
                }
            }
        }
    }
}
