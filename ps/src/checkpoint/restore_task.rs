use anyhow::{bail, Result};
use grpc::sniper::{GpuPsDenseData, GpuPsSparseData};
use util::error_bail;

use std::io::BufRead;

use log::{error, info};
use std::marker::PhantomData;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::dense::DenseVariable;
use crate::embedding::{Embedding, SparseParameter};

use super::file_handler::{FileReader, HdfsFileReader, LocalFileReader};
use super::tool::CheckpointContext;

/// Parse proto from base64 string.
#[inline]
pub fn parse_proto_base64<M: prost::Message + Default>(line: &String) -> Result<M> {
    let s = match STANDARD.decode(line) {
        Ok(x) => x,
        Err(err) => {
            error_bail!("decode line base64 failed! error: {}", err);
        }
    };

    match M::decode(s.as_slice()) {
        Ok(x) => Ok(x),
        Err(err) => {
            error_bail!("SimpleFeaturs parse proto failed! error: {}", err);
        }
    }
}

/// Restore sparse embedding parameters from file.
///
/// It's easy to change `R` to hdfs file reader or other reader.
pub struct RestoreSparseTask<R: FileReader> {
    /// Context parameters for different task.
    pub context: CheckpointContext,

    /// W is not used in field, only used when `run` is executed, so must use `PhantomData` to mark it.
    marker: PhantomData<R>,
}

impl<R: FileReader> RestoreSparseTask<R> {
    pub fn new(context: &CheckpointContext) -> Self {
        Self {
            context: context.clone(),
            marker: PhantomData,
        }
    }

    #[inline]
    fn is_weight_file(&self, filename: &String) -> bool {
        filename.ends_with(".weight")
    }

    #[inline]
    fn is_adagrad_file(&self, filename: &String) -> bool {
        filename.ends_with(".adagrad")
    }

    #[inline]
    fn has_nan(&self, values: &[f32]) -> bool {
        values.iter().any(|x| x.is_nan())
    }

    /// Helper function for restore sparse parameters.
    ///
    /// Use `is_embedding_weight` to distinguish embedding parameters and optimizer parameters.
    fn restore_sparse_parameters(
        &self,
        filename: &String,
        embedding: &Embedding,
        is_embedding_weight: bool,
    ) -> Result<()> {
        let reader = R::get_reader(filename)?;

        let mut total_count: u64 = 0;

        let embedding_size = embedding.embedding_size as usize;

        for x in reader.lines() {
            match x {
                Ok(line) => {
                    let sparse_data = parse_proto_base64::<GpuPsSparseData>(&line)?;

                    // Check id len and val len.
                    if sparse_data.id.len() * embedding_size != sparse_data.val.len() {
                        error_bail!(
                            "id.len() * embedding_size != val.len() for sparse parameter, id.len(): {}, val.len(): {}, varname: {}, filename: {}",
                            sparse_data.id.len(),
                            sparse_data.val.len(),
                            self.context.varname.clone(),
                            filename.clone(),
                        );
                    }

                    for (i, sign) in sparse_data.id.iter().enumerate() {
                        let start = i * embedding_size;
                        let end = start + embedding_size;

                        let weight = &sparse_data.val[start..end];

                        if self.has_nan(weight) {
                            error!(
                                "sparse parameter has nan! varname: {}, filename: {}, sign: {}, start: {}, end: {}, is_embedding_weight: {}",
                                self.context.varname.clone(),
                                filename.clone(),
                                sign,
                                start,
                                end,
                                is_embedding_weight,
                            );
                            continue;
                        }

                        total_count += 1;

                        match embedding.store.get_mut(&sign) {
                            Some(mut x) => {
                                // If `SparseParameter` exists, write into weight in `SparseParameter`.
                                for (j, v) in weight.iter().enumerate() {
                                    let index = if is_embedding_weight {
                                        j
                                    } else {
                                        j + embedding_size
                                    };

                                    if index < x.weight.len() {
                                        x.weight[index] = *v;
                                    } else {
                                        error_bail!(
                                            "out of range, index: {}, x.weight.len(): {}, is_embedding_weight: {}",
                                            index,
                                            x.weight.len(),
                                            is_embedding_weight,
                                        );
                                    }
                                }
                            }
                            None => {
                                // If `SparseParameter` not exists, insert new `SparseParameter` with weight.
                                //
                                // For adagrad.
                                //
                                // TODO: Add trait to support more optimizer.
                                let new_param =
                                    SparseParameter::with_half_slice(weight, is_embedding_weight);
                                embedding.store.insert(sign.clone(), new_param);
                            }
                        }
                    }
                }
                Err(err) => {
                    error_bail!(
                        "read line failed! varname: {}, path: {}, err: {}",
                        self.context.varname.clone(),
                        self.context.path.clone(),
                        err
                    );
                }
            }
        }

        info!(
            "restore sparse embedding parameters done, varname: {}, filename: {}, total_count: {}",
            self.context.varname.clone(),
            filename.clone(),
            total_count,
        );

        Ok(())
    }

    fn restore_weight(&self, filename: &String, embedding: &Embedding) -> Result<()> {
        self.restore_sparse_parameters(filename, embedding, true)
    }

    fn restore_adagrad(&self, filename: &String, embedding: &Embedding) -> Result<()> {
        self.restore_sparse_parameters(filename, embedding, false)
    }

    /// Restore embedding parameters from file.
    ///
    /// Filename is stored in context.path. Weight filename and adagrad filename are in pair.
    /// We read parameters from file and insert them into embedding store.
    pub fn run(&self, embedding: &Embedding) -> Result<()> {
        let filenames = self
            .context
            .path
            .split(",")
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        for filename in filenames.iter() {
            if self.is_weight_file(filename) {
                self.restore_weight(filename, embedding)?;
            } else if self.is_adagrad_file(filename) {
                self.restore_adagrad(filename, embedding)?;
            } else {
                // Ignore other files.
            }
        }

        Ok(())
    }
}

/// Restore dense parameters from file.
///
/// It's easy to change `R` to hdfs file reader or other reader.
pub struct RestoreDenseTask<R: FileReader> {
    /// Context parameters for different task.
    pub context: CheckpointContext,

    /// W is not used in field, only used when `run` is executed, so must use `PhantomData` to mark it.
    marker: PhantomData<R>,
}

impl<R: FileReader> RestoreDenseTask<R> {
    pub fn new(context: &CheckpointContext) -> Self {
        Self {
            context: context.clone(),
            marker: PhantomData,
        }
    }

    /// Restore dense parameters from file.
    ///
    /// Filename is stored in context.path.
    pub fn run(&self, dense_variable: &mut DenseVariable) -> Result<()> {
        let reader = R::get_reader(&self.context.path)?;

        let mut _total_count: u64 = 0;

        for x in reader.lines() {
            match x {
                Ok(line) => {
                    let dense_data = parse_proto_base64::<GpuPsDenseData>(&line)?;
                    dense_variable
                        .push_from_slice(&dense_data.value, dense_data.offset_idx as usize)?;

                    _total_count += dense_data.value.len() as u64;
                }
                Err(err) => {
                    error_bail!(
                        "read line failed! varname: {}, path: {}, err: {}",
                        self.context.varname.clone(),
                        self.context.path.clone(),
                        err
                    );
                }
            }
        }

        Ok(())
    }
}

pub type RestoreSparseFromLocalTask = RestoreSparseTask<LocalFileReader>;
pub type RestoreDenseFromLocalTask = RestoreDenseTask<LocalFileReader>;

pub type RestoreSparseFromHdfsTask = RestoreSparseTask<HdfsFileReader>;
pub type RestoreDenseFromHdfsTask = RestoreDenseTask<HdfsFileReader>;
