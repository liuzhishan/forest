use anyhow::{bail, Result};
use grpc::sniper::{GpuPsDenseData, GpuPsSparseData};
use std::cmp::min;
use std::io::Write;
use util::error_bail;

use log::{error, info};
use std::marker::PhantomData;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::dense::DenseVariable;
use crate::embedding::Embedding;

use super::file_handler::{FileWriter, HdfsFileWriter, LocalFileWriter};
use super::tool::CheckpointContext;

/// Encode message to base64, then save to one line in file.
#[inline]
pub fn append_proto_base64_to_file<M: prost::Message, W: Write>(
    message: &M,
    writer: &mut W,
) -> Result<()> {
    let mut buf = Vec::new();

    message.encode(&mut buf)?;

    let s = STANDARD.encode(&buf);
    let _ = writer.write(s.as_bytes())?;
    let _ = writer.write("\n".as_bytes())?;
    let _ = writer.flush();

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
    #[inline]
    fn get_filename_prefix(&self) -> String {
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
        info!(
            "save task start, varname: {}, shard_index: {}",
            self.context.varname.clone(),
            self.context.shard_index
        );

        if self.context.inner_shard_total == 0 {
            error_bail!(
                "context.inner_shard_total is 0! varname: {}",
                self.context.varname.clone(),
            );
        }

        let prefix = self.get_filename_prefix();

        let weight_filename = format!("{}.weight", prefix);
        info!("weight_filename: {}", weight_filename.clone());

        let mut weight_writer = W::get_writer(&weight_filename)?;

        let adagrad_filename = format!("{}.adagrad", prefix);
        info!("adagrad_filename: {}", adagrad_filename.clone());

        let mut adagrad_writer = W::get_writer(&adagrad_filename)?;

        let variable_dim = self.context.variable_dim;
        let optimizer_dim = self.context.optimizer_dim;

        let mut weight_sparse = GpuPsSparseData::default();

        weight_sparse
            .id
            .reserve(self.context.max_record_iterate_count as usize);
        weight_sparse
            .val
            .reserve(self.context.max_record_iterate_count as usize * variable_dim);

        let mut adagrad_sparse = GpuPsSparseData::default();
        adagrad_sparse
            .id
            .reserve(self.context.max_record_iterate_count as usize);
        adagrad_sparse
            .val
            .reserve(self.context.max_record_iterate_count as usize * optimizer_dim);

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

                weight_sparse.id.push(x.key().clone());
                weight_sparse
                    .val
                    .extend_from_slice(&weight[0..variable_dim]);

                adagrad_sparse.id.push(x.key().clone());
                adagrad_sparse
                    .val
                    .extend_from_slice(&weight[variable_dim..weight.len()]);

                if weight_sparse.id.len() >= self.context.max_record_iterate_count as usize {
                    // When reach max_record_iterate_count, save one line to file, then clear.
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

/// Save dense to file.
pub struct SaveDenseTask<W: FileWriter> {
    /// Parameters for saving.
    pub context: CheckpointContext,

    /// W is used in `run` function.
    marker: PhantomData<W>,
}

impl<W: FileWriter> SaveDenseTask<W> {
    pub fn new(context: &CheckpointContext) -> Self {
        Self {
            context: context.clone(),
            marker: PhantomData,
        }
    }

    /// Get final filename to write by parameters in context.
    #[inline]
    fn get_final_filename(&self) -> String {
        // Dense varname may have special chars, such as `/`, `:`, we encode the varname to base64 first.
        let varname_base64 = STANDARD.encode(&self.context.varname);

        format!("{}/{}.dense", self.context.path.clone(), varname_base64,)
    }

    pub fn run(&self, dense: &DenseVariable) -> Result<()> {
        // Max size is 4m.
        let max_size = 4_194_304;
        let total = dense.values.len();

        let final_filename = self.get_final_filename();
        let mut writer = W::get_writer(&final_filename)?;

        let mut dense_data = GpuPsDenseData::default();

        dense_data.name = self.context.varname.clone();
        dense_data.value.reserve(max_size);

        let n: usize = total / max_size;

        for i in 0..(n + 1) {
            let start = i * max_size;
            let end = min((i + 1) * max_size, total);

            // Save `max_size` each line.
            dense_data
                .value
                .extend_from_slice(&dense.values[start..end]);

            // Set offset and length. This is important, since restore need the index.
            dense_data.offset_idx = start as i32;
            dense_data.total_length = (end - start + 1) as i32;

            append_proto_base64_to_file(&dense_data, &mut writer)?;

            // Clear for next processing.
            dense_data.value.clear();
        }

        Ok(())
    }
}

pub type SaveSparseToLocalTask = SaveSparseTask<LocalFileWriter>;
pub type SaveDenseToLocalTask = SaveDenseTask<LocalFileWriter>;

pub type SaveSparseToHdfsTask = SaveSparseTask<HdfsFileWriter>;
pub type SaveDenseToHdfsTask = SaveDenseTask<HdfsFileWriter>;
