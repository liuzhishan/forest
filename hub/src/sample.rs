//! Sample for input data in different format.

use anyhow::{bail, Result};
use base64::encode_engine_slice;
use log::{error, info};

use rand::RngCore;
use util::error_bail;

/// SampleBatch contains a batch of input sample.
///
/// For performance consideration, the data is organized by field. For sparse feature,
/// each sign is attached with a batch index. For dense feature, since the size is fixed
/// for each dense feature, we can just store the floats in the array one by one.
///
/// For example:
/// field0: [1, 2, 3, 4]
/// index0: [0, 0, 1, 2]
///
/// There are several reasons why this format is good for performance:
/// 1. Parameter in ps is organized by field, which means feature index. When hub second
/// data to ps, it needs to be send by field. So organized by field is easy for the communication
/// with ps.
/// 2. Sparse feature signs organized by field is good for futher performance optimization,
/// such as simd acceleration and compression.
#[derive(Debug, Clone)]
pub struct SampleBatch {
    /// Batch size.
    pub batch_size: usize,

    /// Sparse feature count. Organize by field.
    pub sparse_feature_count: usize,

    /// Dense feature count.
    pub dense_feature_count: usize,

    /// Sparse feature signs.
    pub sparse_signs: Vec<Vec<u64>>,

    /// Item index, must be align with sparse signs.
    pub item_indexes: Vec<Vec<usize>>,

    /// Dense features. Organize by batch_index.
    pub dense_features: Vec<Vec<Vec<f32>>>,

    /// Labels. Organized by batch_index.
    pub labels: Vec<i32>,

    /// batch_id, random u64.
    pub batch_id: u64,
}

impl SampleBatch {
    pub fn new(batch_size: usize, sparse_feature_count: usize, dense_feature_count: usize) -> Self {
        let mut sparse_signs = Vec::with_capacity(sparse_feature_count);
        sparse_signs.resize(sparse_feature_count, Vec::new());

        let mut item_indexes = Vec::with_capacity(sparse_feature_count);
        item_indexes.resize(sparse_feature_count, Vec::new());

        let mut dense_features = Vec::with_capacity(batch_size);
        dense_features.resize(batch_size, Vec::new());

        for i in 0..dense_features.len() {
            dense_features[i].resize(dense_feature_count, Vec::new());
        }

        let mut labels = Vec::new();
        labels.resize(batch_size, 0);

        let batch_id: u64 = rand::thread_rng().next_u64();

        Self {
            batch_size,
            sparse_feature_count,
            dense_feature_count,
            sparse_signs,
            item_indexes,
            dense_features,
            labels,
            batch_id,
        }
    }

    /// Add sparse features.
    ///
    /// Sign may come from protobuf, or other vecs.
    pub fn add_sparse_feature(
        &mut self,
        batch_index: usize,
        field: usize,
        signs: &[u64],
    ) -> Result<()> {
        if field >= self.sparse_signs.len() {
            error_bail!(
                "out of range, field: {}, sparse_signs.len(): {}",
                field,
                self.sparse_signs.len()
            );
        }

        if field >= self.item_indexes.len() {
            error_bail!(
                "out of range, field: {}, item_indexes.len(): {}",
                field,
                self.item_indexes.len()
            );
        }

        if batch_index >= self.batch_size {
            error_bail!(
                "out of range, batch_index: {}, batch_size: {}",
                batch_index,
                self.batch_size,
            );
        }

        self.sparse_signs[field].extend_from_slice(signs);

        for _ in 0..signs.len() {
            self.item_indexes[field].push(batch_index);
        }

        Ok(())
    }

    /// Add dense features.
    ///
    /// Values are float list, may come from protobuf, or other vecs.
    pub fn add_dense_feature(
        &mut self,
        batch_index: usize,
        field: usize,
        values: &[f32],
    ) -> Result<()> {
        if batch_index >= self.dense_features.len() {
            error_bail!(
                "out of range, batch_index: {}, dense_features.len(): {}",
                batch_index,
                self.dense_features.len()
            );
        }

        if field >= self.dense_features[batch_index].len() {
            error_bail!(
                "out of range, field: {}, dense_features[batch_index].len(): {}, batch_index: {}",
                field,
                self.dense_features[batch_index].len(),
                batch_index
            );
        }

        self.dense_features[batch_index][field].extend_from_slice(values);

        Ok(())
    }

    /// Add label.
    pub fn add_label(&mut self, batch_index: usize, label: i32) -> Result<()> {
        if batch_index >= self.labels.len() {
            error_bail!(
                "out of range, batch_index: {}, labels.len(): {}",
                batch_index,
                self.labels.len()
            );
        }

        self.labels[batch_index] = label;

        Ok(())
    }
}
