use anyhow::{anyhow, bail, Result};
use log::{error, info};

use tokio::time::{sleep, Duration};
use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};
use util::{error_bail, FeaturePlacement};

use crate::{batch_assembler::BatchAssembler, feed_sample::FeedSample, hdfs_reader::HdfsReader};

use super::local_reader::LocalReader;

use super::sample::SampleBatch;
use grpc::sniper::{SimpleFeatures, StartSampleOption};

/// Pipeline for SingleSample input data.
///
/// Read single sample line by line from input data and assembly batch, then send to ps and trainer.
pub struct SingleSamplePipeline {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Sender for SamppleBatch. Receiver is used for ReadSample.
    sample_batch_sender: async_channel::Sender<SampleBatch>,

    /// Queue size for str line.
    line_queue_size: usize,

    /// Queue size for SimpleFeatures.
    feature_queue_size: usize,

    /// Queue size for sample batch.
    sample_batch_queue_size: usize,
}

impl SingleSamplePipeline {
    pub fn new(
        option: StartSampleOption,
        sample_batch_sender: async_channel::Sender<SampleBatch>,
    ) -> Self {
        Self {
            option,
            sample_batch_sender,
            line_queue_size: 2048,
            feature_queue_size: 2048,
            sample_batch_queue_size: 100,
        }
    }

    pub async fn init(&mut self) -> bool {
        // TODO
        true
    }

    fn get_filenames_from_option(&self) -> Vec<String> {
        let mut res = Vec::<String>::new();

        // Use hdfs src parameter temperary, change to local src later.
        match &self.option.hdfs_src {
            Some(hdfs_src) => hdfs_src.file_list.iter().for_each(|filename| {
                res.push(format!("{}/{}", hdfs_src.dir.clone(), filename.clone()));
            }),
            None => {}
        }

        res
    }

    /// Start the pipeline for SimpleFeatures input.
    ///
    /// Create LocalReader, BatchAssembler, FeedSample task, and channel for each task,
    /// init the tasks, and then run the tasks.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        info!("SingleSamplePipeline start");

        let (line_sender, line_receiver) = async_channel::bounded::<String>(self.line_queue_size);

        let (batch_sender, batch_receiver) =
            async_channel::bounded::<SampleBatch>(self.sample_batch_queue_size);

        info!("SingleSamplePipeline after get channel");

        // Create LocalReader.
        let filenames = self.get_filenames_from_option();
        let mut hdfs_reader = HdfsReader::new(&filenames, line_sender);

        if !hdfs_reader.init().await {
            error_bail!(
                "hdfs_reader init failed! filenames: {}",
                filenames.join(", ")
            );
        }

        info!("SingleSamplePipeline after local_reader init");

        let batch_size = self.option.batch_size as usize;

        if self.option.feature_list.is_none() {
            error_bail!("feature_list in StartSampleOption is None!");
        }

        let feature_list = self.option.feature_list.as_ref().unwrap().clone();

        let sparse_feature_count = feature_list.sparse_class_names.len() as usize;
        let dense_feature_count = feature_list.dense_class_names.len() as usize;

        let dense_total_size = self.option.dense_total_size as usize;

        // Create BatchAssembler.
        let mut batch_assembler = BatchAssembler::new(
            batch_size,
            sparse_feature_count,
            dense_feature_count,
            dense_total_size,
            line_receiver,
            batch_sender,
        );

        if !batch_assembler.init().await {
            error_bail!("batch_assembly init failed!");
        }

        info!("SingleSamplePipeline after batch assembler init");

        // Create FeedSample.
        info!(
            "start sample, emb_tables: {}, ps_eps: {}",
            self.option
                .emb_tables
                .iter()
                .map(|x| x.name.clone())
                .collect::<Vec<_>>()
                .join(", "),
            self.option.ps_eps.join(", "),
        );

        let placement = FeaturePlacement::new(&self.option.emb_tables, &self.option.ps_eps);
        let ps_endpoints = &self.option.ps_eps;

        let mut feed_sample = FeedSample::new(
            self.option.clone(),
            batch_receiver,
            placement,
            self.sample_batch_sender.clone(),
            ps_endpoints,
        );

        if !feed_sample.init().await {
            error_bail!("feed_sample init failed!");
        }

        info!("SingleSamplePipeline after init");

        // Run the tasks.
        subsys.start(SubsystemBuilder::new("hdfs_reader", |a| {
            hdfs_reader.run(a)
        }));

        subsys.start(SubsystemBuilder::new("batch_assembler", |a| {
            batch_assembler.run(a)
        }));

        subsys.start(SubsystemBuilder::new("feed_sample", |a| feed_sample.run(a)));

        info!("SingleSamplePipeline start subsys done");

        subsys.on_shutdown_requested().await;

        info!("SingleSamplePipeline shutdown!");

        Ok(())
    }
}

/// Pipeline for GroupSample input data.
///
/// Read group sample from input data, then send to ps and trainer.
pub struct GroupSamplePipeline {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Sender for SamppleBatch. Receiver is used for ReadSample.
    sample_batch_sender: async_channel::Sender<SampleBatch>,
}

impl GroupSamplePipeline {
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

    /// Start the pipeline for SimpleFeatures input.
    ///
    /// Create StartSample, LocalReader, BatchAssembler, FeedSample task, and channel for each task,
    /// init the tasks, and then run the tasks.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        // TODO
        Ok(())
    }
}
