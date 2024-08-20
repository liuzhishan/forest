use grpc::sniper::Role;
use hashbrown::HashMap;

use std::{
    borrow::Borrow,
    sync::{Mutex, OnceLock},
};

use crossbeam_utils::CachePadded;

use crate::{
    checkpoint::checkpoint_manager::CheckpointManager,
    embedding::Embedding,
    scheduler::Scheduler,
    variable_manager::{DenseManager, EmbeddingManager},
};

use dashmap::{mapref::one::Ref, RawRwLock};

/// Global env for all ps workers.
///
/// Embedding_manager, dense_manager, scheduler and other important state are all managed by EnvImpl.
#[derive(Default)]
pub struct Env {
    /// Role.
    pub role: i32,

    // Role id.
    pub role_id: i32,

    /// schduler ps.
    pub scheduler_ps: String,

    /// Model name.
    pub model_name: String,

    /// is_freezed.
    pub is_freezed: bool,

    /// Embedding learning rate.
    pub embedding_lr: f32,

    /// Embedding eps.
    pub embedding_eps: f32,

    /// Max feed queue size for embedding variable.
    pub max_feed_queue_size: u64,

    /// Max lookup queue size for embedding variable.
    pub max_lookup_queue_size: u64,
}

impl Env {
    pub fn new() -> Self {
        let mut env = Self::default();

        env.max_feed_queue_size = 1024;
        env.max_lookup_queue_size = 1024;

        env
    }
}
