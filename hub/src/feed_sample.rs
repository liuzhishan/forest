use anyhow::{anyhow, bail, Result};
use grpc::sniper::StartSampleOption;
use log::{error, info};

use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};
use util::{error_bail, FeaturePlacement};

use super::sample::SampleBatch;
use grpc::sniper::sniper_ps_client::SniperPsClient;
use grpc::sniper::{sniper_hub_client::SniperHubClient, FeedSampleOption};

/// Send features to ps and trainer.
///
/// For each SampleBatch, first generate a random u64 as batch_id, then send batch_id and sparse
/// features to ps, and send batch_id, dense_features and labels to trainer.
pub struct FeedSample {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Receiver from BatchAssembler or BatchReader.
    sample_batch_receiver: async_channel::Receiver<SampleBatch>,

    /// SniperHubClient for grpc request.
    // ps_client: SniperPsClient<tonic::transport::Channel>,

    /// FeaturePlacement.
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
        // TODO
        true
    }

    /// Send batch_id and sparse features to ps.
    pub async fn send_to_ps(&mut self, sample_batch: &SampleBatch) -> Result<()> {
        // TODO
        info!("batch: {:?}", sample_batch.clone());
        Ok(())
    }

    /// Send batch_id, dense features and labels to trainer.
    pub async fn send_to_trainer(&mut self, sample_batch: &SampleBatch) -> Result<()> {
        // TODO
        Ok(())
    }

    /// Process the sample batch, send feature to ps and trainer.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        loop {
            tokio::select! {
                sample_batch_res = self.sample_batch_receiver.recv() => {
                    match sample_batch_res {
                        Ok(sample_batch) => {
                            self.total_batch += 1;

                            info!(
                                "get one batch! total_batch: {}, batch_id: {}",
                                self.total_batch,
                                sample_batch.batch_id
                            );

                            self.send_to_ps(&sample_batch).await?;
                            self.send_to_trainer(&sample_batch).await?;
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
