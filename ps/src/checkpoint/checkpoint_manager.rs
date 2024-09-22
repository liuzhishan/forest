use std::default;

use anyhow::{anyhow, bail, Result};
use grpc::sniper::sniper_client::SniperClient;
use log::{error, info};

use dashmap::DashMap;
use dashmap::Map;
use dashmap::SharedValue;
use dashmap::{mapref::one::Ref, RawRwLock};

/// Restore state.
#[derive(Default, Clone)]
pub struct RestoreState {
    /// Total file count need to restore.
    pub total_file_count: u32,

    /// Success count.
    pub success_count: u32,

    /// Whether the restore is already failed.
    pub has_error: bool,
}

impl RestoreState {
    /// Construct RestoreState with `total_file_count`.
    pub fn new(total_file_count: u32, has_error: bool) -> Self {
        Self {
            total_file_count,
            success_count: 0,
            has_error,
        }
    }

    /// If it's success.
    #[inline]
    pub fn is_success(&self) -> bool {
        self.success_count == self.total_file_count
    }

    /// Increment the success count.
    #[inline]
    pub fn increment_success(&mut self, count: u32) {
        self.success_count += count;
    }
}

/// Manage checkpoint related info, such as restore and save state.
pub struct CheckpointManager {
    /// Restore state of each variable.
    ///
    /// The key is a string combined with varname and shard index, and the value is success
    /// file count. The restoring is considered success only if the success file count is
    /// equal to number of files in `restore_option`. So we only need to track whether each
    /// varname is restored successfully.
    ///
    /// We use `i32` valeu to represent different state.
    /// 0: init state for each varname and shard index.
    /// -1: failed.
    /// i32 which is > 0: success count, if the count is equal to total file count, then it's
    ///     success, otherwise need to wait.
    var_restore_state: DashMap<String, RestoreState>,
}

impl CheckpointManager {
    pub fn new() -> Self {
        Self {
            var_restore_state: DashMap::new(),
        }
    }

    /// Generate the restore key from varname and shard_index.
    #[inline]
    fn get_restore_key(&self, varname: &String, shard_index: i32) -> String {
        format!("{}_:_{}", varname.clone(), shard_index)
    }

    /// Whether the var is in `var_restore_state`.
    #[inline]
    pub fn contains_var_shard(&self, varname: &String, shard_index: i32) -> bool {
        let key = self.get_restore_key(varname, shard_index);
        self.var_restore_state.contains_key(&key)
    }

    /// Record to `restore_state`.
    pub fn insert_restore_state(
        &mut self,
        varname: &String,
        shard_index: i32,
        file_count: u32,
        is_success: bool,
    ) -> Result<()> {
        let key = self.get_restore_key(varname, shard_index);

        match self.var_restore_state.get_mut(&key) {
            Some(mut value) => {
                if is_success {
                    value.increment_success(file_count);
                } else {
                    value.has_error = true;
                }

                Ok(())
            }
            None => Err(anyhow!(
                "cannot find restore state, varname: {}, shard_index: {}",
                varname.clone(),
                shard_index
            )),
        }
    }

    /// Set each varname shard value to 0.
    #[inline]
    pub fn init_restore_state(&mut self, varname: &String, shard_index: i32, file_count: u32) {
        let key = self.get_restore_key(varname, shard_index);
        match self.var_restore_state.get(&key) {
            Some(_) => {}
            None => {
                self.var_restore_state
                    .insert(key, RestoreState::new(file_count, false));
            }
        }
    }

    /// Get varname restore state.
    #[inline]
    pub fn get_restore_state(&self, varname: &String, shard_index: i32) -> Option<RestoreState> {
        let key = self.get_restore_key(varname, shard_index);

        match self.var_restore_state.get(&key) {
            Some(v) => Some(v.value().clone()),
            None => None,
        }
    }
}
