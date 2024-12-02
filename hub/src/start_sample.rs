use anyhow::Result;

use tokio_graceful_shutdown::SubsystemHandle;

use grpc::sniper::{SimpleFeatures, StartSampleOption};

use crate::sample::SampleBatch;

/// Start read single sample.
pub struct StartSingleSample {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Send SimpleFeatures to next task.
    features_sender: async_channel::Sender<SimpleFeatures>,
}

impl StartSingleSample {
    pub fn new(
        option: StartSampleOption,
        features_sender: async_channel::Sender<SimpleFeatures>,
    ) -> Self {
        Self {
            option,
            features_sender,
        }
    }

    pub async fn init(&mut self) -> bool {
        // TODO
        true
    }

    /// Process the data.
    pub async fn run(self, _subsys: SubsystemHandle) -> Result<()> {
        // TODO
        Ok(())
    }
}

/// Start read group Sample.
pub struct StartGroupSample {
    // StartSampleOption, passed from trainer.
    option: StartSampleOption,

    /// Send SimpleFeatures to next task.
    sample_batch_sender: async_channel::Sender<SampleBatch>,
}

impl StartGroupSample {
    pub fn new(
        option: StartSampleOption,
        sample_batch_sender: async_channel::Sender<SampleBatch>,
    ) -> Self {
        Self {
            option,
            sample_batch_sender,
        }
    }

    pub async fn init(&mut self) -> bool {
        // TODO
        true
    }

    /// Process the data.
    pub async fn run(self, _subsys: SubsystemHandle) -> Result<()> {
        // TODO
        Ok(())
    }
}
