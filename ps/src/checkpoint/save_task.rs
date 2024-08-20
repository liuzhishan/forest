use anyhow::{anyhow, bail, Result};
use grpc::sniper::{GpuPsSparseData, VariableType};
use util::error_bail;
use std::{fs::File, io::Write};

use std::marker::PhantomData;
use log::{info, error};

use base64::Engine;
use base64::{engine::general_purpose::STANDARD, read::DecoderReader};

use crate::embedding::Embedding;
use crate::env::Env;

use super::tool::CheckpointContext;

/// Trait for write to file.
///
/// Maybe local file or hdfs file.
pub trait FileWriter: Sized {
    /// Open a file by filename.
    fn new(filename: &String) -> Result<Self>;

    /// Write content in buf, return the total bytes.
    fn write(&mut self, buf: &[u8]) -> Result<usize>;

    /// Flush the content in buffer to target.
    fn flush(&mut self) -> Result<()>;
}

pub fn append_proto_base64_to_file<M: prost::Message, W: FileWriter>(
    message: &M,
    writer: &mut W,
) -> Result<()> {
    let mut buf = Vec::new();

    message.encode(&mut buf)?;

    let s = STANDARD.encode(&buf);
    let total = writer.write(s.as_bytes())?;

    Ok(())
}

/// Save sparse embedding parameters to file.
///
/// Embedding is passing as a reference to SaveSparseTask.
///
/// It's easy to change W to hdfs file writer or other writer.
pub struct SaveSparseTask<W: FileWriter> {
    /// Context parameters for different task.
    pub context: CheckpointContext,

    /// W is not used in field, only used when `run` is executed, so must use `PhantomData` to mark it.
    marker: PhantomData<W>,
}

impl<W: FileWriter> SaveSparseTask<W> {
    pub fn new(context: &CheckpointContext) -> Self {
        Self {
            context: context.clone(),
            marker: PhantomData,
        }
    }

    /// Get final filename to write by parameters in context.
    fn get_prefix(&self) -> String {
        format!(
            "{}/{}.{}_{}",
            self.context.path.clone(),
            self.context.varname.clone(),
            self.context.shard_index,
            self.context.inner_shard,
        )
    }

    /// Save sparse parameters.
    ///
    /// Filename is determined by parameters in context.
    ///
    /// Each task save a part of all signs. Since the order of `DashMap` keys is not determined, we need
    /// a way to split all signs into groups ahead, and pass parameter to each task. We can use the modular
    /// of the sign and total to split all signs into bucket. Total is computed before dispatching task, by
    /// total sign count and `context.max_record_iterate_count`.
    pub fn run(&self, embedding: &Embedding) -> Result<()> {
        if self.context.inner_shard_total == 0 {
            error_bail!(
                "context.inner_shard_total is 0! varname: {}",
                self.context.varname.clone(),
            );
        }

        let prefix = self.get_prefix();

        let weight_filename = format!("{}.weight", prefix);
        let mut weight_writer = W::new(&weight_filename)?;

        let adagrad_filename = format!("{}.adagrad", prefix);
        let mut adagrad_writer = W::new(&adagrad_filename)?;

        /// Count total signs get.
        let mut total_count: i64 = 0;

        let variable_dim = self.context.variable_dim;
        let optimizer_dim = self.context.optimizer_dim;

        let mut weight_sparse = GpuPsSparseData::default();

        weight_sparse.id.reserve(self.context.max_record_iterate_count as usize);
        weight_sparse.val.reserve(self.context.max_record_iterate_count as usize * variable_dim);

        let mut adagrad_sparse = GpuPsSparseData::default();
        adagrad_sparse.id.reserve(self.context.max_record_iterate_count as usize);
        adagrad_sparse.val.reserve(self.context.max_record_iterate_count as usize * optimizer_dim);

        let inner_shard = self.context.inner_shard as u64;
        let inner_shard_total = self.context.inner_shard_total as u64;

        for x in embedding.store.iter() {
            // Only handle key which falls into the bucket of inner_shard.
            if x.key() % inner_shard_total == inner_shard {
                let weight = &x.value().weight;

                // weight.len() must be equal to variable_dim + optimizer_dim.
                if weight.len() != variable_dim + optimizer_dim {
                    error_bail!(
                        "weight.len() != variable_dim + optimizer_dim, weight.len(): {}, variable_dim: {}, optimizer_dim: {}, varname: {}",
                        weight.len(),
                        variable_dim,
                        optimizer_dim,
                        self.context.varname.clone(),
                    );
                }

                total_count += 1;

                weight_sparse.id.push(x.key().clone());
                weight_sparse.val.extend_from_slice(&weight[0..variable_dim]);

                adagrad_sparse.id.push(x.key().clone());
                adagrad_sparse.val.extend_from_slice(&weight[variable_dim..weight.len()]);

                if weight_sparse.id.len() >= self.context.max_record_iterate_count as usize {
                    // When reach max_record_iterate_count, save one line to file, then clear.
                    let buf: Vec<u8> = Vec::new();

                    // Save to file.
                    append_proto_base64_to_file(&weight_sparse, &mut weight_writer)?;
                    append_proto_base64_to_file(&adagrad_sparse, &mut adagrad_writer)?;

                    // Clear for later process.
                    weight_sparse.id.clear();
                    weight_sparse.val.clear();

                    adagrad_sparse.id.clear();
                    adagrad_sparse.val.clear();
                }
            }
        }

        // The remain parameters.
        append_proto_base64_to_file(&weight_sparse, &mut weight_writer)?;
        append_proto_base64_to_file(&adagrad_sparse, &mut adagrad_writer)?;

        info!(
            "save sparse shard done, varname: {}, shard_index: {}, shard_num: {}, inner_shard: {}, inner_shard_total: {}",
            self.context.varname.clone(),
            self.context.shard_index,
            self.context.shard_num,
            inner_shard,
            inner_shard_total,
        );

        Ok(())
    }
}

/// Write to local file.
pub struct LocalFileWriter {
    /// filename to write.
    filename: String,
    writer: File,
}

impl FileWriter for LocalFileWriter {
    fn new(filename: &String) -> Result<Self> {
        match File::create(filename) {
            Ok(writer) => Ok(Self {
                filename: filename.clone(),
                writer,
            }),
            Err(err) => Err(err.into()),
        }
    }

    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        match self.writer.write(buf) {
            Ok(x) => Ok(x),
            Err(err) => Err(err.into()),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self.writer.flush() {
            Ok(_) => Ok(()),
            Err(err) => Err(err.into()),
        }
    }
}
