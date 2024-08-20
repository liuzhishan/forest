use grpc::sniper::{CheckPointTarget, CheckPointType, VariableType};
use tonic_types::ErrorDetail;
use util::Status;

/// Important info used for save or restore checkpoint.
///
/// For example, save to hdfs, or restore from hdfs. Task should use the right parameter for diffrent task.
#[derive(Default, Clone)]
pub struct CheckpointContext {
    /// Version of checkpoint format.
    pub version: i64,

    /// CheckPointType, such as `incr`, `full`. Only `full` is supported now,
    /// `incr` will be supported in the future.
    pub checkpoint_type: CheckPointType,

    /// CheckPointTarget, such as `hdfs`, `local`.
    pub checkpoint_target: CheckPointTarget,

    /// Path. For `hdfs` path is `hdfs` path, for `local` path is local filename.
    pub path: String,

    /// Model name. Should be globally unique.
    pub model_name: String,

    /// Varname to be saved.
    pub varname: String,

    /// Sparse or dense.
    pub variable_type: VariableType,

    /// Shard index of sparse embedding.
    pub shard_index: i32,

    /// Total shard number of current sparse embedding var.
    pub shard_num: i32,

    /// Start sign index for sparse embedding var when iterating signs.
    pub start: usize,

    /// End sign index for sparse embedding var when iterating signs.
    pub end: usize,

    /// Whether should we wait the task to be finished.
    pub need_finished: bool,

    /// Whether the task has finished.
    pub has_finished: bool,

    /// Inner shard of one shard_index, to distinguish different task or thread.
    pub inner_shard: i32,

    /// Inner shard total, how many shard should one embedding should be split into in one ps.
    pub inner_shard_total: usize,

    /// Max iteration count of record for saving parameters.
    pub max_record_iterate_count: i32,

    /// Dim of variable embedding parameters.
    pub variable_dim: usize,

    /// Dim of optimizer parameters. It's same as variable_dim for adagrad.
    pub optimizer_dim: usize,
}

/// Result of checkpoint related task.
#[derive(Default, Clone)]
pub struct CheckpointResult {
    /// Status of different task result.
    pub status: Status,

    /// Total record count.
    pub record_count: usize,

    /// Total bytes count.
    pub bytes_count: usize,

    /// Time spend in milliseconds.
    pub time_cost_in_ms: i64,
}
