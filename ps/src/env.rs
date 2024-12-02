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

    /// Max feed queue size for embedding variable. Default is 1024.
    pub max_feed_queue_size: usize,

    /// Max lookup queue size for embedding variable. Default is 1024.
    pub max_lookup_queue_size: usize,

    /// Max save key size in one file. Default is 10_000_000.
    pub max_save_key_size_in_file: u64,
}

impl Env {
    pub fn new() -> Self {
        let mut env = Self::default();

        env.max_feed_queue_size = 1024;
        env.max_lookup_queue_size = 1024;

        env.max_save_key_size_in_file = 10_000_000;

        env
    }
}
