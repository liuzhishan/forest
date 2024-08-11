use anyhow::{anyhow, bail, Result};
use log::{error, info};

use tokio::time::{sleep, Duration};

use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};

use prost::Message;

use base64::Engine;
use base64::{engine::general_purpose::STANDARD, read::DecoderReader};

use grpc::sniper::SimpleFeatures;
use util::error_bail;

use super::sample::SampleBatch;

/// Assembly the data into batch.
///
/// The batch size must be provided in construction of BatchAssembler. BatchAssembler read
/// samples from stream, and assembly samples to SampleBatch.
pub struct BatchAssembler {
    /// Batch size.
    batch_size: usize,

    /// Sparse feature count.
    pub sparse_feature_count: usize,

    /// Dense feature count.
    pub dense_feature_count: usize,

    /// Dense total size.
    pub dense_total_size: usize,

    /// Current items count in buffer.
    batch_index: usize,

    /// Sample batch.
    sample_batch: SampleBatch,

    /// Strings read from input data. Each line is a single sample.
    line_receiver: async_channel::Receiver<String>,

    /// SampleBatch sender after assembly.
    batch_sender: async_channel::Sender<SampleBatch>,

    /// Total line received.
    total_line: i64,

    /// Total batch.
    total_batch: i64,
}

impl BatchAssembler {
    pub fn new(
        batch_size: usize,
        sparse_feature_count: usize,
        dense_feature_count: usize,
        dense_total_size: usize,
        line_receiver: async_channel::Receiver<String>,
        batch_sender: async_channel::Sender<SampleBatch>,
    ) -> Self {
        let sample_batch = SampleBatch::new(
            batch_size,
            sparse_feature_count,
            dense_feature_count,
            dense_total_size,
        );

        Self {
            batch_size,
            sparse_feature_count,
            dense_feature_count,
            dense_total_size,
            batch_index: 0,
            sample_batch,
            line_receiver,
            batch_sender,
            total_line: 0,
            total_batch: 0,
        }
    }

    pub fn init(&mut self) -> bool {
        true
    }

    /// Assembly the features read from input to batch, and increment the batch_index by 1.
    fn assembly(&mut self, features: &SimpleFeatures) -> Result<()> {
        // add sparse features.
        for (i, sparse) in features.sparse_feature.iter().enumerate() {
            self.sample_batch
                .add_sparse_feature(self.batch_index, i, &sparse.values)?;
        }

        // add dense features.
        for (i, dense) in features.dense_feature.iter().enumerate() {
            self.sample_batch
                .add_dense_feature(self.batch_index, i, &dense.values)?;
        }

        // add labels.
        self.sample_batch
            .add_label(self.batch_index, features.label as i32)?;

        self.batch_index += 1;

        Ok(())
    }

    /// Process the input features, and assmbly features to batch, then send to the
    /// batch_sender channel.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        loop {
            tokio::select! {
                line_res = self.line_receiver.recv() => {
                    if line_res.is_err() {
                        // Need to sleep, if the channel is closed, all other channel will be finish
                        // and closed too. Then all task is done, shutdonw signal is send, and the
                        // all pipeline is finished before FeedSample is over.
                        //
                        // So the task will run forever, can only be closed by shutdown signal from
                        // outside the pipeline.
                        //
                        // Need to find another more elegant way later.
                        sleep(Duration::from_secs(2)).await;
                    } else {
                        let line = line_res.unwrap();

                        self.total_line += 1;

                        let s = match STANDARD.decode(line) {
                            Ok(x) => x,
                            Err(err) => {
                                error_bail!("decode line base64 failed! error: {}", err);
                            }
                        };

                        let features = match SimpleFeatures::decode(s.as_slice()) {
                            Ok(x) => x,
                            Err(err) => {
                                error_bail!("SimpleFeaturs parse proto failed! error: {}", err);
                            }
                        };

                        match self.assembly(&features) {
                            Ok(_) => {},
                            Err(err) => {
                                error_bail!("assembly features failed! error: {}", err);
                            }
                        }

                        info!(
                            "[BatchAssembler.run] after assembly feature, batch_index: {}, batch_size: {}, total_line: {}, total_batch: {}",
                            self.batch_index,
                            self.batch_size,
                            self.total_line,
                            self.total_batch,
                        );

                        // If batch_index >= batch_size, we have enough data to send.
                        // Then reset the batch.
                        let is_enough = self.batch_index >= self.batch_size;
                        if is_enough {
                            let new_batch = self.sample_batch;
                            match self.batch_sender.send(new_batch).await {
                                Ok(_) => {},
                                Err(err) => {
                                    error!("send batch failed! error: {}", err);
                                }
                            }

                            info!("[BatchAssembler.run] after send batch");

                            self.total_batch += 1;

                            // After each SampleBatch, must reset batch_index to 0, and sample_batch to new SampleBatch.
                            self.batch_index = 0;
                            self.sample_batch = SampleBatch::new(
                                self.batch_size,
                                self.sparse_feature_count,
                                self.dense_feature_count,
                                self.dense_total_size,
                            );
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("BatchAssembler shutdown!");
                    return Ok(());
                }
            }
        }
    }
}

/// Read batch data from input data.
pub struct BatchReader {}
