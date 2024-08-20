use anyhow::{anyhow, bail, Result};
use dashmap::mapref::one::{Ref, RefMut};
use log::{error, info};

use dashmap::DashMap;

use crate::dense::DenseVariable;

use crate::embedding::Embedding;

/// All embedding variable manager.
///
/// use the embedding varname as the key, Embedding as the value.
///
/// For example: embedding_0 is the Embedding of field 0.
#[derive(Default)]
pub struct EmbeddingManager {
    vars: DashMap<String, Embedding>,
}

impl EmbeddingManager {
    /// Add a new Embedding, must provide the parameter for Embedding.
    pub fn add_new_var(
        &self,
        varname: &String,
        embedding_size: usize,
        shard_num: usize,
        shard_index: usize,
        fields: &Vec<i32>,
        capacity: u64,
        hash_size: usize,
        max_feed_queue_size: u64,
        max_lookup_queue_size: u64,
    ) {
        let embedding = Embedding::new(
            varname,
            embedding_size,
            shard_num,
            shard_index,
            fields,
            capacity,
            hash_size,
            max_feed_queue_size,
            max_lookup_queue_size,
        );

        self.vars.insert(varname.clone(), embedding);
    }

    pub fn get(&self, varname: &String) -> Option<Ref<'_, String, Embedding>> {
        self.vars.get(varname)
    }

    pub fn get_mut(&self, varname: &String) -> Option<RefMut<'_, String, Embedding>> {
        self.vars.get_mut(varname)
    }

    pub fn remove(&self, varname: &String) -> Option<(String, Embedding)> {
        self.vars.remove(varname)
    }
}

/// All dense variable manager.
#[derive(Default)]
pub struct DenseManager {
    vars: DashMap<String, DenseVariable>,
}

impl DenseManager {
    pub fn add_new_var(&self, varname: &String, dims: &Vec<usize>) {
        let dense = DenseVariable::new(varname, dims);
        self.vars.insert(varname.clone(), dense);
    }

    pub fn get(&self, varname: &String) -> Option<Ref<'_, String, DenseVariable>> {
        self.vars.get(varname)
    }

    pub fn get_mut(&self, varname: &String) -> Option<RefMut<'_, String, DenseVariable>> {
        self.vars.get_mut(varname)
    }

    pub fn remove(&self, varname: &String) -> Option<(String, DenseVariable)> {
        self.vars.remove(varname)
    }
}
