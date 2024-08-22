//! Parameter for sparse feature signs.
//!
//! For performance reason, we use `dashmap::DashMap` as the default contrainer to store
//! embedding parameters.
//!
//! The key of `dashmap::DashMap` is the sign of sparse features, and the value is
//! a struct which contains important data for a sign, such as embedding parameter,
//! grad parameter, bucket_size, and so on.
//!
//! There are several containers could be used as embedding table, so we define a trait
//! `EmbeddingTable`. `Embedding` use the trait `EmbeddingTable` as inner store, so other
//! containers can be used through generic parameter.
//!
//! We support two type of signs: hash sign and nohash sign.
//!
//! For hash sign, the origin sign of sparse feature is hashed first, then modulo the
//! bucket_size. The result will be the key of hashmap. We can use `dashmap::DashMap`
//! to store the parameters. Since the bucket_size is fixed, there is another alternative
//! way to store the parameters, which is using a `Vec`. It has better performance than
//! using a `hashmap`.
//!
//! For nohash sign, the key is the origin sign of sparse feature, which is an `u64`.
//! And we cannot known how many keys will be in the map beforehand, it is possible
//! that the size of map will be too large to fit into the memory of `ps`. So we must
//! use `LRUCache` to store the parameters.
//!
use grpc::sniper::EmbeddingLookupOption;
use log::{error, info};
use std::hash::RandomState;

use anyhow::{anyhow, bail, Result};
use dashmap::iter::Iter;
use grpc::sniper::FeedSampleOption;
use grpc::sniper::GpuPsFeature64;
use grpc::sniper::PullOption;
use grpc::sniper::PushOption;

use prost::Message;
use rand::Rng;
use rand::RngCore;

use dashmap::DashMap;
use util::error_bail;
use util::get_target_shard_by_sign;

/// Sparse Parameter for sparse signs.
pub struct SparseParameter {
    /// Weight of sparse sign. Contains both weight and grad parameters.
    /// Initialized with normal distribution.
    pub weight: Vec<f32>,

    /// Whether the parameter is visited.
    pub is_visited: bool,
}

impl SparseParameter {
    pub fn new(embedding_size: usize) -> Self {
        let total = embedding_size * 2;

        let mut weight = Vec::with_capacity(total);

        // Inialized with random normal distribution.
        for i in 0..total {
            weight.push(rand::thread_rng().gen::<f32>());
        }

        Self {
            weight,
            is_visited: false,
        }
    }

    #[inline]
    pub fn with_weight(weight: Vec<f32>) -> Self {
        Self {
            weight,
            is_visited: false,
        }
    }

    #[inline]
    pub fn with_embedding_weight(embedding_weight: Vec<f32>) -> Self {
        Self::with_embedding_weight_slice(embedding_weight.as_slice())
    }

    #[inline]
    pub fn with_optimizer_weight(optimizer_weight: Vec<f32>) -> Self {
        Self::with_optimizer_weight_slice(&optimizer_weight)
    }

    /// Construct SparseParameter from embedding weight of optimizer weight.
    #[inline]
    pub fn with_embedding_weight_slice(embedding_weight: &[f32]) -> Self {
        Self::with_half_slice(embedding_weight, true)
    }

    /// Construct SparseParameter from optimizer weight of optimizer weight.
    #[inline]
    pub fn with_optimizer_weight_slice(optimizer_weight: &[f32]) -> Self {
        Self::with_half_slice(optimizer_weight, false)
    }

    /// Construct SparseParameter from embedding weight or optimizer weight, use `is_embedding_weight`
    /// to distinguish.
    pub fn with_half_slice(param: &[f32], is_embedding_weight: bool) -> Self {
        let embedding_size = param.len();

        let mut weight = vec![0.0; embedding_size * 2];

        for i in 0..embedding_size {
            if is_embedding_weight {
                weight[i] = param[i];
            } else {
                weight[i + embedding_size] = param[i];
            }
        }

        Self {
            weight,
            is_visited: false,
        }
    }
}

/// EmbeddingTable trait for further extensiblity.
///
/// Preallocting memory could be used to impprove performance.
pub trait EmbeddingTable: Default {}

/// Store sparse signs and item indexes in a batch, which is sent from hub.
struct BatchMessage {
    /// batch_id.
    pub batch_id: u64,

    /// Sparse signs in current field.
    pub signs: Vec<u64>,

    /// Item indexes in current batch and current field.
    pub item_indexes: Vec<i32>,
}

/// Result of EmbeddingLookup.
#[derive(Default)]
pub struct EmbeddingLookupResult {
    /// Embedding sum of a batch.
    pub values: Vec<Vec<f32>>,

    /// Time spend in milliseconds.
    pub time_spends: u64,
}

/// Embedding parameters.
pub struct Embedding {
    /// Embedding varname.
    pub varname: String,

    /// Embedding size.
    pub embedding_size: usize,

    /// Shard num of one sparse feature, default is 1.
    pub shard_num: usize,

    /// Shard index of current Embedding.
    pub shard_index: usize,

    /// Capacity of store, used for LRU.
    pub capacity: u64,

    /// Hash size.
    pub hash_size: usize,

    /// Storage of the SparseParameter.
    pub store: DashMap<u64, SparseParameter>,

    /// Feed queue.
    ///
    /// One Embedding may has multi field, the key of outer DashMap if field, the key of inner DashMap
    /// is batch_id.
    feed_queue: DashMap<i32, DashMap<u64, BatchMessage>>,

    /// Max feed_queue size.
    max_feed_queue_size: u64,

    /// Lookup queue.
    ///
    /// One Embedding may has multi field, the key of outer DashMap if field, the key of inner DashMap
    /// is batch_id.
    lookup_queue: DashMap<i32, DashMap<u64, BatchMessage>>,

    /// Max lookup queue.
    max_lookup_queue_size: u64,

    /// Max pull key count for each request. Fixed in new.
    max_pull_key_count: usize,

    /// Fields.
    pub fields: Vec<i32>,
}

impl Embedding {
    /// Construct a new Embedding.
    pub fn new(
        varname: &String,
        embedding_size: usize,
        shard_num: usize,
        shard_index: usize,
        fields: &Vec<i32>,
        capacity: u64,
        hash_size: usize,
        max_feed_queue_size: u64,
        max_lookup_queue_size: u64,
    ) -> Self {
        let feed_queue = DashMap::new();
        let lookup_queue = DashMap::new();

        for field in fields {
            feed_queue.insert(field.clone(), DashMap::new());
            lookup_queue.insert(field.clone(), DashMap::new());
        }

        Self {
            varname: varname.clone(),
            embedding_size,
            store: DashMap::new(),
            shard_num,
            shard_index,
            capacity,
            hash_size,
            feed_queue,
            max_feed_queue_size,
            lookup_queue,
            max_lookup_queue_size,
            max_pull_key_count: 100000,
            fields: fields.clone(),
        }
    }

    /// Read the signs and item_indexes in feed_sample_option, and put them in feed_queue.
    ///
    /// The data is encoded in proto message GpuPsFeature64, need to be deserialize first.
    /// Then copy the data into BatchMessage, and put into feed_queue.
    pub fn feed_sample(
        &self,
        batch_id: u64,
        feed_sample_option: &FeedSampleOption,
        info_index: usize,
    ) -> Result<()> {
        let field_info_opt = feed_sample_option.field_info.get(info_index);
        if field_info_opt.is_none() {
            error_bail!(
                "out of range, info_index: {}, field_info.len(): {}",
                info_index,
                feed_sample_option.field_info.len(),
            );
        }

        let field_info = field_info_opt.as_ref().unwrap();

        // Deserialize from proto bytes.
        let feature_bytes = &field_info.feature;
        let feature_pb = match GpuPsFeature64::decode(feature_bytes.as_slice()) {
            Ok(x) => x,
            Err(err) => {
                error_bail!("decode GpuPsFeature64 failed! error: {}", err,);
            }
        };

        let batch_message = BatchMessage {
            batch_id,
            signs: feature_pb.features,
            item_indexes: feature_pb.item_indices,
        };

        let field = field_info.field_idx;

        self.push_feed_queue(field, batch_message)
    }

    /// Put one BatchMessage to feed_queue, use field as the outer key, batch_id as the inner key.
    fn push_feed_queue(&self, field: i32, batch_message: BatchMessage) -> Result<()> {
        let mut inner_opt = self.feed_queue.get_mut(&field);
        if inner_opt.is_none() {
            error_bail!("cannot find inner feed queue, field: {}", field,);
        }

        let mut inner = inner_opt.unwrap();

        let batch_id = batch_message.batch_id;
        inner.insert(batch_id, batch_message);

        // If exceed the max_queue_size, need to delete some BatchMessage.
        // Although in hashed model, this would not happen. So right now, just delete the first item
        // for simplicity.
        //
        // TODO: It should be done by LRU cache for general case, will be fixed later.
        if inner.len() as u64 > self.max_feed_queue_size {
            let cnt = (inner.len() as u64 - self.max_feed_queue_size) as usize;

            let keys = inner
                .iter()
                .map(|x| x.key().clone())
                .take(cnt)
                .collect::<Vec<_>>();

            for key in keys.iter() {
                inner.remove(&key);
            }
        }

        Ok(())
    }

    /// Get BatchMessage by field and batch_id
    fn pull_feed_queue(&self, field: i32, batch_id: u64) -> Option<BatchMessage> {
        match self.feed_queue.get_mut(&field) {
            Some(inner) => match inner.remove(&batch_id) {
                Some(x) => Some(x.1),
                None => {
                    error!("cannot find batch, batch_id: {} field: {}", batch_id, field);
                    None
                }
            },
            None => {
                error!("cannot find field in feed_queue, field: {}", field);
                None
            }
        }
    }

    /// Push sign and corresponding parameter into store.
    ///
    /// The parameter contains both embedding parameter and grad parameter, right now just copy the
    /// float values into store.
    ///
    /// Different optimizer can be supported through well-defined trait. Support for different optimizer
    /// will be added later.
    pub fn push(
        &self,
        batch_id: u64,
        option: &PushOption,
        keys: &[u64],
        values: &[f32],
    ) -> Result<()> {
        let total = self.embedding_size * 2;

        for i in 0..keys.len() {
            let key = keys[i];

            // skip sign which is not in current ps.
            if get_target_shard_by_sign(key, self.shard_num) != self.shard_index {
                continue;
            }

            let pos = i * total;

            match self.store.get_mut(&key) {
                Some(mut v) => {
                    if v.weight.len() < total {
                        v.weight.resize(total, 0.0);
                    }

                    for i in 0..total {
                        if pos + i < values.len() {
                            v.weight[i] = values[pos + i];
                        } else {
                            error!(
                                "out of range, index: {}, values.len(): {}",
                                pos + i,
                                values.len()
                            );
                        }
                    }
                }
                None => {
                    let weight = values[pos..pos + total].to_vec();
                    let sparse_parameter = SparseParameter::with_weight(weight);
                    self.store.insert(key.clone(), sparse_parameter);
                }
            }
        }

        Ok(())
    }

    /// Reset all `is_visited` in store value to false.
    fn reset_all_visited(&self) {
        self.store.iter_mut().for_each(|mut x| {
            x.value_mut().is_visited = false;
        });
    }

    /// Get all signs and parameters from store.
    ///
    /// The number of signs maybe big, so one call may not be able to get all parameters.
    /// We need to call multiple times from trainer, and each time get `max_pull_key_count` signs.
    /// There will be only one client request in only one trainer. So the different request is handled
    /// on by one.
    ///
    /// We need to keep track of the iterator progress.
    ///
    /// There are four ideas to solve this problem.
    ///
    /// 1. The easiest way is just record the total count already pulled, and for each request, we loop over
    /// the store map from start, and skip the item already handled, and stop the loop when we reach the
    /// `max_pull_key_count` for each request. Although this method is not effecient, since we need to loop
    /// over almost the entire map for each request. There is also another problem, the order of each
    /// iteration is different. We also need to keep track of which key is already visited. We can add a
    /// bool value `is_visited` to SparseParameter to indicate whether visited. When first request arrived,
    /// we need to set all `is_visited` to false.
    ///
    /// 2. Add an iter type as field of Embedding. Then use this iter to loop over the store map. When the
    /// count exceeds `max_pull_key_count` for one request, we return the result, and next request starts from
    /// the same postion where last request stop. So for each request, we get at most `max_pull_key_count`
    /// parameters. But when the iterator is used, it's consumed, so we cannot use it next time.
    ///
    /// 3. Implement fast forward for iterator of store map. For each request, we record the total keys pulled.
    /// And before loop over the map, we try to fast forward the iterator to the position we want. If the iterator
    /// of store map can be converted to slice, then we can forward the iterator fast using slice iterator, then
    /// convert back to map iter. I'm Not sure whether iterator of `DashMap` can be converted to slice, Need to
    /// checkout the detail of `DashMap`.
    ///
    /// 4. Preallocate enough memory, and use the memory to be storage of `HashMap`. This approach give us the
    /// power of manually management the memory. When we delete an item from `HashMap`, we just mark it as deleted,
    /// not actually deallocate it. So the next item can reuse the memory. When iterating, we can jump to the
    /// target position using index in just O(1) time. It will be much more effecient because there are no runtime
    /// memory allocating and deacllocating, and fast forward of iterator. But it's also harder to implement.
    ///
    /// Considering the performance and implementation, We use the first approach for the first version. The
    /// fourth approach will be supported in the future.
    ///
    /// TODO: using preallocated memory and DashMap as embedding table storage.
    pub fn pull(
        &self,
        batch_id: u64,
        option: &PullOption,
        out_option: &mut PullOption,
        keys: &mut Vec<u64>,
        values: &mut Vec<f32>,
    ) {
        // First request, we need to set all `is_visited` to `false` for all keys.
        if option.progress == 0 {
            self.reset_all_visited();
        }

        let mut total: i64 = 0;

        for mut x in self.store.iter_mut() {
            if !x.is_visited {
                // Copy the keys and values.
                keys.push(x.key().clone());
                values.extend_from_slice(x.value().weight.as_slice());

                // Mark the key has been visited.
                x.is_visited = true;

                total += 1;
            }

            if total >= self.max_pull_key_count as i64 {
                break;
            }
        }

        out_option.progress = option.progress + total;
        out_option.completed = out_option.progress == self.store.len() as i64;
    }

    /// Get embedding lookup result for field and batch_id, sum the weights of signs.
    ///
    /// The result is a one-dimension tensor, the total length is batch_size * embedding_size.
    /// The content is sum of weight of each item concat together.
    ///
    /// For example:
    ///
    /// batch_size = 2, embedding_size = 4, the result is:
    /// [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.89, 0.9]
    ///
    /// Find signs by batch_id first, then get embedding parameter from store by the signs.
    /// Sum the weight for each item_index, then return the result.
    pub fn embedding_lookup(
        &self,
        field: i32,
        batch_id: u64,
        batch_size: usize,
    ) -> Result<EmbeddingLookupResult> {
        let mut res = EmbeddingLookupResult::default();

        // Initialize with 0.0.
        res.values.resize(batch_size, Vec::new());
        for i in 0..res.values.len() {
            res.values[i].resize(self.embedding_size, 0.0);
        }

        // get batch_message received from hub.
        let batch_message = match self.pull_feed_queue(field, batch_id) {
            Some(x) => x,
            None => {
                error_bail!(
                    "cannot find batch_message, field: {}, batch_id: {}",
                    field,
                    batch_id,
                );
            }
        };

        if batch_message.signs.len() != batch_message.item_indexes.len() {
            error_bail!("signs.len() != item_indexes.len() in batch_message!");
        }

        let signs = &batch_message.signs;
        let item_indexes = &batch_message.item_indexes;

        let total = signs.len();

        // get embedding parameter by sign and sum.
        for i in 0..total {
            let item_index = item_indexes[i] as usize;

            if item_index > res.values.len() {
                error_bail!(
                    "out of range, item_index: {}, res.values.len(): {}",
                    item_index,
                    res.values.len(),
                );
            }

            match self.store.get(&signs[i]) {
                Some(x) => {
                    // If found sign, sum the embedding weight to res.values.
                    for j in 0..self.embedding_size {
                        res.values[item_index][j] += x.weight[j];
                    }
                }
                None => {
                    // If not found, insert new SparseParameter into store.
                    self.store
                        .insert(signs[i], SparseParameter::new(self.embedding_size));
                }
            }
        }

        // After lookup, push batch_message to lookup_queue for grad parameter updating later.
        self.push_lookup_queue(field, batch_id, batch_message)?;

        Ok(res)
    }

    /// Update the gradient parameters to store.
    ///
    /// Only support AdaGrad optimizer now. Need trait to support other optimizer. It will be supported in the future.
    ///
    /// TODO: Flexible trait to support more optimizer.
    pub fn push_grad(
        &self,
        grad: &[f32],
        batch_id: u64,
        field: i32,
        learning_rate: f32,
        eta: f32,
        eps: f32,
        decay: f32,
        l2: f32,
    ) -> Result<()> {
        let batch_message = match self.pull_lookup_queue(field, batch_id) {
            Some(x) => x,
            None => {
                error_bail!(
                    "cannot find batch_message in lookup_queue, field: {}, batch_id: {}",
                    field,
                    batch_id,
                );
            }
        };

        let signs = &batch_message.signs;
        let item_indexes = &batch_message.item_indexes;

        for sign in signs {
            match self.store.get_mut(sign) {
                Some(mut x) => {
                    // For each sign, sum the grad parameter to weight.
                    self.apply_adagrad_w(
                        &mut x.weight,
                        grad,
                        self.embedding_size,
                        learning_rate,
                        eta,
                        eps,
                        decay,
                        l2,
                    )?;
                }
                None => {
                    // Skip. Maybe need some log.
                }
            }
        }

        Ok(())
    }

    /// Sum the grad parameter to weight.
    ///
    /// Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    ///
    /// It could be speed up using simd operation.
    ///
    /// TODO: use simd to speed up.
    fn apply_adagrad_w(
        &self,
        weight: &mut Vec<f32>,
        grad: &[f32],
        embedding_size: usize,
        learning_rate: f32,
        eta: f32,
        eps: f32,
        decay: f32,
        l2: f32,
    ) -> Result<()> {
        if weight.len() != grad.len() * 2 {
            error_bail!(
                "weight.len() != grad.len() * 2, weight.len(): {}, grad.len(): {}",
                weight.len(),
                grad.len(),
            );
        }

        if grad.len() != embedding_size {
            error_bail!(
                "grad.len() != embedding_size, grad.len(): {}, embedding_size: {}",
                grad.len(),
                embedding_size,
            );
        }

        // The first half of weight if embedding parameter, the second half of weight is grad parameter.
        for i in 0..embedding_size {
            let g: f32 = weight[i] + l2 * weight[i];

            // index of grad parameter.
            let j = embedding_size + i;

            weight[j] = weight[j] + g * g;
            weight[i] = weight[i] - weight[i] * decay - g * eta / (eps + weight[j]).sqrt();
        }

        Ok(())
    }

    /// Push batch_message to lookup queue, for grad parameter updating.
    fn push_lookup_queue(
        &self,
        field: i32,
        batch_id: u64,
        batch_message: BatchMessage,
    ) -> Result<()> {
        match self.lookup_queue.get(&field) {
            Some(inner) => {
                // Move match_message into inner lookup queue.
                inner.insert(batch_id, batch_message);

                // Delete more batch_message when inner queue is full.
                if inner.len() >= self.max_lookup_queue_size as usize {
                    let cnt = inner.len() - self.max_lookup_queue_size as usize;

                    let keys = inner
                        .iter()
                        .map(|x| x.key().clone())
                        .take(cnt)
                        .collect::<Vec<_>>();
                    keys.iter().for_each(|x| {
                        inner.remove(x);
                    })
                }

                Ok(())
            }
            None => {
                error_bail!(
                    "cannot find field in lookup_queue when push_lookup_queue, field: {}",
                    field
                );
            }
        }
    }

    fn pull_lookup_queue(&self, field: i32, batch_id: u64) -> Option<BatchMessage> {
        match self.lookup_queue.get(&field) {
            Some(inner) => match inner.remove(&batch_id) {
                Some(x) => Some(x.1),
                None => {
                    error!(
                        "cannot find batch_id in inner when pull_lookup_queue, field: {}, batch_id: {}",
                        field,
                        batch_id,
                    );
                    None
                }
            },
            None => {
                error!(
                    "cannot find field in lookup_queue when pull_lookup_queue, field: {}",
                    field,
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dashmap::DashMap;
    use log::{error, info};
    use std::collections::hash_map::RandomState;
    use std::sync::Once;

    use util::{error_bail, init_log, Flags};

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            init_log();
        });
    }

    #[test]
    fn test_map_iter() {
        setup();

        info!("test_map_view for dashmap");

        let dm: DashMap<i32, i32> = DashMap::new();

        dm.insert(0, 0);
        dm.insert(4, 4);
        dm.insert(9, 9);
        dm.insert(12, 12);

        let iter = dm.iter();

        for x in iter {
            info!("key: {}, value: {}", x.key(), x.value());
        }
    }
}
