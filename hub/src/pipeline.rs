use anyhow::{anyhow, bail, Result};
use log::{error, info};

use tokio::sync::{broadcast, mpsc};
use tokio::time::{sleep, Duration};
use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};

use util::histogram::Histogram;
use util::{
    error_bail,
    histogram::{HistogramDetail, HistogramType},
    FeaturePlacement,
};

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

    /// Histogram statistics.
    histogram: Histogram,

    /// Sender for ps shard.
    ///
    /// Why use sender instead of receiver?
    ///
    /// Because we need to create ps shard task in new thread, and `broadcast` need
    /// all receivers to consume the data, so all receivers must be in use. If we
    /// use receiver here, the receiver will be never used. So we need to create
    /// receiver from `sender.subscribe()` to avoid this problem.
    ps_shard_sender: broadcast::Sender<Vec<Vec<i32>>>,
}

impl SingleSamplePipeline {
    pub fn new(
        option: StartSampleOption,
        sample_batch_sender: async_channel::Sender<SampleBatch>,
        histogram: Histogram,
        ps_shard_sender: broadcast::Sender<Vec<Vec<i32>>>,
    ) -> Self {
        Self {
            option,
            sample_batch_sender,
            line_queue_size: 2048,
            feature_queue_size: 2048,
            sample_batch_queue_size: 100,
            histogram,
            ps_shard_sender,
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
                if hdfs_src.dir.len() > 0 {
                    res.push(format!("{}/{}", hdfs_src.dir.clone(), filename.clone()));
                } else {
                    res.push(filename.clone());
                }
            }),
            None => {}
        }

        res
    }

    /// Split filenames between spawned tasks.
    fn get_part_filenames(&self, filenames: &Vec<String>, index: i32, total: i32) -> Vec<String> {
        let n = filenames.len();
        filenames
            .iter()
            .enumerate()
            .filter(|(i, x)| *i as i32 % total == index)
            .map(|(i, x)| x.clone())
            .collect::<Vec<String>>()
    }

    /// Start the pipeline for SimpleFeatures input.
    ///
    /// Create LocalReader, BatchAssembler, FeedSample task, and channel for each task,
    /// init the tasks, and then run the tasks.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        info!("SingleSamplePipeline start");

        // Run the tasks.

        let parallel = self.option.parallel;

        let (line_sender, line_receiver) = async_channel::bounded::<String>(self.line_queue_size);

        let (batch_sender, batch_receiver) =
            async_channel::bounded::<SampleBatch>(self.sample_batch_queue_size);

        info!("SingleSamplePipeline after init");

        let filenames = self.get_filenames_from_option();

        // Create HdfsReader.
        let n = parallel / 2;
        info!("filenames.len(): {}, n: {}", filenames.len(), n);

        for i in 0..n {
            let part_filenames = self.get_part_filenames(&filenames, i, n);

            let mut hdfs_reader = HdfsReader::new(
                &part_filenames,
                line_sender.clone(),
                self.histogram.clone(),
                i,
            );

            if !hdfs_reader.init().await {
                error_bail!(
                    "hdfs_reader init failed! filenames: {}",
                    filenames.join(", ")
                );
            }

            let hdfs_reader_name = format!("hdfs_reader_{}", i);
            subsys.start(SubsystemBuilder::new(hdfs_reader_name, |a| {
                hdfs_reader.run(a)
            }));
        }

        let m = parallel / 4;
        for i in 0..m {
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
                line_receiver.clone(),
                batch_sender.clone(),
                self.histogram.clone(),
            );

            if !batch_assembler.init().await {
                error_bail!("batch_assembly init failed!");
            }

            let batch_assembler_name = format!("batch_assembler_{}", i);
            subsys.start(SubsystemBuilder::new(batch_assembler_name, |a| {
                batch_assembler.run(a)
            }));

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
                batch_receiver.clone(),
                placement,
                self.sample_batch_sender.clone(),
                ps_endpoints,
                self.histogram.clone(),
                self.ps_shard_sender.subscribe(),
            );

            if !feed_sample.init().await {
                error_bail!("feed_sample init failed!");
            }

            let feed_sample_name = format!("feed_sample_{}", i);
            subsys.start(SubsystemBuilder::new(feed_sample_name, |a| {
                feed_sample.run(a)
            }));
        }
        info!("SingleSamplePipeline start subsys done");

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

    /// Histogram statistics.
    histogram: Histogram,

    /// Sender for ps shard.
    ps_shard_sender: broadcast::Sender<Vec<Vec<i32>>>,
}

impl GroupSamplePipeline {
    pub fn new(
        option: StartSampleOption,
        sample_batch_sender: async_channel::Sender<SampleBatch>,
        histogram: Histogram,
        ps_shard_sender: broadcast::Sender<Vec<Vec<i32>>>,
    ) -> Self {
        Self {
            option,
            sample_batch_sender,
            histogram,
            ps_shard_sender,
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
