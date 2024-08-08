use anyhow::{anyhow, bail, Result};
use grpc::sniper::{FeedFieldInfo, StartSampleOption};
use log::{error, info};

use hashbrown::HashMap;

use ps::get_ps_client;
use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};
use util::{error_bail, FeaturePlacement};

use super::sample::SampleBatch;
use grpc::sniper::sniper_ps_client::SniperPsClient;
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

    /// SniperHubClient for grpc request.
    // ps_client: SniperPsClient<tonic::transport::Channel>,

    /// FeaturePlacement is used to determine which ps each var live in.
    feature_placement: FeaturePlacement,

    /// Total batch.
    total_batch: i64,
}

impl FeedSample {
    pub fn new(
        option: StartSampleOption,
        sample_batch_receiver: async_channel::Receiver<SampleBatch>,
        feature_placement: FeaturePlacement,
    ) -> Self {
        Self {
            option,
            sample_batch_receiver,
            feature_placement,
            total_batch: 0,
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

        let feature_list = self.option.feature_list.unwrap();

        // For extensability, the value of sparse_features is a HashMap of String
        // and FeedSampleOption. Each key of the inner map is ps_endpoint. Right now
        // there are only one ps_endpoint for each sparse embedding table.
        //
        // EmbeddingTable -> ps_endpoint -> FeedSamleOption
        let sparse_features: HashMap<String, HashMap<String, FeedSampleOption>> = HashMap::new();

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
            let shard_endpoints = self.feature_placement.get_emb_placement(varname);

            if shard_endpoints.size() == 0 {
                error_bail!(
                    "cannot get shard_endpoints for varname: {}",
                    varname.clone()
                );
            }

            // Get all data send to ps.
            let one_features: HashMap<String, FeedSampleOption> = HashMap::new();

            // Assembly FeedSampleOption.
            let mut feed_sample_option = FeedSampleOption::default();
            feed_sample_option.batch_size = sample_batch.batch_size as u32;
            feed_sample_option.work_mode = self.option.work_mode;

            // Assembly field_info.
            let mut field_info = FeedFieldInfo::default();

            field_info.field_idx = i as i32;
            field_info.field_dim = feature_list.sparse_field_count[i] as i32;
        }

        // Write signs to FeedFieldInfo in FeedSampleOption.
        Ok(())
    }

    /// Send batch_id, dense features and labels to trainer.
    pub async fn send_to_trainer(&mut self, sample_batch: SampleBatch) -> Result<()> {
        // TODO
        Ok(())
    }

    /// Process the sample batch, send feature to ps and trainer.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        // Each thread get a ps_client.
        let mut ps_client = get_ps_client().await?;

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

                            self.send_to_ps(&sample_batch, &mut ps_client).await?;
                            self.send_to_trainer(sample_batch).await?;
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
