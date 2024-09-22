use anyhow::{anyhow, bail, Result};
use dashmap::{DashMap, DashSet};
use log::{error, info};
use util::error_bail;
use std::time::Duration;

use std::{collections::VecDeque, sync::atomic::AtomicBool};
use tokio_graceful_shutdown::{SubsystemBuilder, Toplevel};
use tokio_graceful_shutdown::SubsystemHandle;

use tokio::{select, sync::mpsc};

use grpc::sniper::{
    heartbeat_option::{CkpStatus, StatusType},
    sniper_client::SniperClient,
    CheckPointTarget, CheckPointType, HeartbeatOption, Role, TensorMessage,
};
use hashbrown::HashMap;

use crate::{checkpoint::tool::CheckpointContext, get_ps_client, request_handler::RwLock};

/// Checkpoint version info.
#[derive(Default)]
pub struct CheckpointVersionInfo {
    /// Checkpoint target, such as hdfs.
    pub checkpoint_target: CheckPointTarget,

    /// Count success or task.
    pub count_success: i32,

    /// Model path.
    pub model_path: String,
}

impl CheckpointVersionInfo {
    pub fn new(
        checkpoint_target: CheckPointTarget,
        count_success: i32,
        model_path: &String,
    ) -> Self {
        Self {
            checkpoint_target,
            count_success,
            model_path: model_path.clone(),
        }
    }
}

#[derive(Default)]
pub struct VersionSuccess {
    pub version: i64,
    pub is_success: bool,
}

/// Finish record of checkpoint versions.
///
/// For performance reasons, and the version info is coming in order by time, we use VecDeque
/// to store the success status of each version.
pub struct FinishRecord {
    /// All records for finished versions.
    full_records: VecDeque<VersionSuccess>,

    /// Max records size to in VecDeque. If exceed, we pop the oldest record.
    max_records_size: usize,
}

impl Default for FinishRecord {
    fn default() -> Self {
        Self {
            full_records: VecDeque::default(),
            max_records_size: 100,
        }
    }
}

impl FinishRecord {
    /// Add a new record to full_records.
    ///
    /// If the record is already exists, update `is_success` field.
    /// Otherwise, push a new `VersionSuccess` record into full_records.
    pub fn add_record(
        &mut self,
        version: i64,
        is_success: bool,
        checkpoint_target: CheckPointTarget,
        checkpoint_type: CheckPointType,
    ) -> Result<()> {
        if checkpoint_target == CheckPointTarget::CkpTargetNfs
            && checkpoint_type == CheckPointType::CkpTypeFull
        {
            let mut is_found = false;

            // If found, update `is_success`.
            for record in self.full_records.iter_mut() {
                if record.version == version {
                    record.is_success = is_success;
                    is_found = true;
                    break;
                }
            }

            // If not found, push new `VersionSuccess` into self.full_records.
            if !is_found {
                self.full_records.push_back(VersionSuccess {
                    version,
                    is_success,
                });

                // Since we only care about the recent records, if full_records size is bigger
                // than max_records_size, we need to remove the old records.
                while self.full_records.len() > self.max_records_size {
                    self.full_records.pop_front();
                }
            }

            Ok(())
        } else {
            Err(anyhow!(format!(
                "unsupported checkpoint_type: {}",
                checkpoint_type.as_str_name().clone()
            )))
        }
    }

    /// Check the result of version, return whether it's `finish` and `is_success`.
    pub fn check_record(&self, version: i64) -> (bool, bool) {
        let mut is_finished = false;
        let mut is_success = false;

        for record in self.full_records.iter() {
            if record.version == version {
                is_finished = true;
                is_success = record.is_success;
            }
        }

        (is_finished, is_finished)
    }
}

/// Send save state to scheduler ps.
pub struct SaveStateNotifier {
    /// Scheduler ps.
    scheduler_ps: String,

    /// Receiver of save state channel.
    receiver: mpsc::Receiver<(CheckpointContext, bool)>,
}

impl SaveStateNotifier {
    pub fn new(
        scheduler_ps: &String,
        receiver: mpsc::Receiver<(CheckpointContext, bool)>,
    ) -> Self {
        Self {
            scheduler_ps: scheduler_ps.clone(),
            receiver,
        }
    }

    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        let mut client = match get_ps_client(&self.scheduler_ps).await {
            Ok(client) => {
                client
            }
            Err(err) => {
                error!("get ps client failed!");
                return Err(err.into());
            }
        };

        loop {
            select! {
                x = self.receiver.recv() => {
                    match x {
                        Some((checkpoint_context, is_success)) => {
                            let mut option = HeartbeatOption::default();

                            option.st_type = StatusType::TypeCkp.into();

                            let mut ckp_st = CkpStatus::default();
                            ckp_st.incr = checkpoint_context.checkpoint_type == CheckPointType::CkpTypeIncr;
                            ckp_st.version = checkpoint_context.version;
                            ckp_st.var_name = checkpoint_context.varname.clone();
                            ckp_st.success = is_success;
                            ckp_st.ckp_target = checkpoint_context.checkpoint_target.into();
                            ckp_st.nfs_path = checkpoint_context.path.clone();
                            ckp_st.shard_idx = checkpoint_context.shard_index;
                            ckp_st.shard_num = checkpoint_context.shard_num;

                            option.ckp_st.insert(ckp_st);

                            let request =
                                TensorMessage::with_option(Role::Ps, &checkpoint_context.varname, &mut option)?;

                            client.heartbeat(request).await;
                        },
                        None => {
                            error!("save state sender is closed!");
                            return Err(anyhow!("save state sender is closed!"));
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("scheduler save state sender shutdown!");
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}

#[derive(Default)]
pub struct Scheduler {
    /// Scheduler ps. There is only one scheculer_ps.
    scheduler_ps: String,

    /// Sender for save state.
    pub save_state_sender: Option<mpsc::Sender<(CheckpointContext, bool)>>,

    /// Hub endpoints.
    hub_endpoints: Vec<String>,

    /// Ps endpoints.
    ps_endpoints: Vec<String>,

    /// Dense vars.
    dense_vars: HashMap<String, String>,

    /// Sparse vars.
    sparse_vars: HashMap<String, String>,

    /// Record finished versions.
    finish_records: RwLock<FinishRecord>,

    /// Full version info.
    full_version_info: DashMap<i64, CheckpointVersionInfo>,

    /// Full version stat.
    full_version_stat: DashMap<i64, DashMap<String, DashSet<String>>>,

    /// Give every varname an empty stat set.
    empty_stat: DashMap<String, DashSet<String>>,

    /// Ps shard of each varname.
    ps_shard: HashMap<String, Vec<String>>,

    /// Total sparse shard of all features.
    total_sparse_shard: i32,
}

impl Scheduler {
    pub fn init(
        &mut self,
        scheduler_ps: &String,
        hub_endpoints: &Vec<String>,
        ps_endpoints: &Vec<String>,
        dense_vars: &HashMap<String, String>,
        sparse_vars: &HashMap<String, String>,
        ps_shard: &HashMap<String, Vec<String>>,
    ) -> bool {
        self.scheduler_ps = scheduler_ps.clone();

        let (sender, receiver) = mpsc::channel(100);

        self.save_state_sender.insert(sender);

        Self::start_save_state_receiver(scheduler_ps.clone(), receiver);

        self.hub_endpoints = hub_endpoints.clone();
        self.ps_endpoints = ps_endpoints.clone();
        self.dense_vars = dense_vars.clone();
        self.sparse_vars = sparse_vars.clone();
        self.ps_shard = ps_shard.clone();

        self.total_sparse_shard = 0;

        for x in ps_shard.iter() {
            self.total_sparse_shard += x.1.len() as i32;

            self.empty_stat.insert(x.0.clone(), DashSet::new());
        }

        for y in dense_vars.iter() {
            self.empty_stat.insert(y.0.clone(), DashSet::new());
        }

        true
    }

    /// Send save state to channel.
    pub async fn send_save_task_state(
        &self,
        checkpoint_context: CheckpointContext,
        is_success: bool,
    ) -> Result<()> {
        match self.save_state_sender.as_ref() {
            Some(sender) => {
                sender.send((checkpoint_context, is_success)).await?;
                Ok(())
            },
            None => {
                Err(anyhow!("no save state sender found!"))
            }
        }
    }

    /// Start a new thread to send save state to scheduler ps.
    ///
    /// Receive checkpoint context and success flag from the sender.
    /// And assembly the `TensorMessage` request, then send to scheduler ps using `heartbeat` request.
    fn start_save_state_receiver(
        scheduler_ps: String,
        receiver: mpsc::Receiver<(CheckpointContext, bool)>,
    ) {
        let notifier = SaveStateNotifier::new(&scheduler_ps, receiver);

        tokio::spawn(async move {
            Toplevel::new(|s| async move {
                s.start(SubsystemBuilder::new("save_state_receiver", |a| {
                    notifier.run(a)
                }));
            })
            .catch_signals()
            .handle_shutdown_requests(Duration::from_millis(1000))
            .await
        });
    }

    pub fn start(&mut self) {}

    pub fn stop(&mut self) {}

    /// Maintain status of each checkpoint task after task is done.
    ///
    /// Update or remove version info based on parameters.
    pub fn post_checkpoint_status(
        &mut self,
        is_increment: bool,
        version: i64,
        varname: &String,
        is_success: bool,
        checkpoint_target: CheckPointTarget,
        path: &String,
        shard_index: i32,
        shard_num: i32,
    ) -> Result<()> {
        let checkpoint_type = if is_increment {
            CheckPointType::CkpTypeIncr
        } else {
            CheckPointType::CkpTypeFull
        };

        let shard_index_str = shard_index.to_string();

        if is_success {
            // If success, update `full_version_stat` and `full_records`.
            match self.full_version_stat.get_mut(&version) {
                Some(x) => match x.get(varname) {
                    Some(a) => {
                        a.insert(shard_index_str);
                    }
                    None => {
                        let shards: DashSet<String> = DashSet::new();
                        shards.insert(shard_index_str);

                        x.insert(varname.clone(), shards);
                    }
                },
                None => {
                    let shards: DashSet<String> = DashSet::new();
                    shards.insert(shard_index_str);

                    let stats = self.empty_stat.clone();
                    stats.insert(varname.clone(), shards);

                    self.full_version_stat.insert(version.clone(), stats);
                }
            }

            match self.full_version_info.get_mut(&version) {
                Some(mut x) => {
                    x.value_mut().count_success += 1;
                }
                None => {
                    let version_info = CheckpointVersionInfo::new(checkpoint_target, 1, path);

                    self.full_version_info.insert(version.clone(), version_info);
                }
            }

            Ok(())
        } else {
            // If not success, remove version from `full_version_stat`.
            self.full_version_stat.remove(&version);

            self.finish_records.write().add_record(
                version,
                is_success,
                checkpoint_target,
                checkpoint_type,
            )?;

            Ok(())
        }
    }

    /// Check whether checkpoint is finished and success.
    ///
    /// Right now only `CheckPointTarget::CkpTargetNfs` is supported.
    pub fn check_checkpoint_status(
        &self,
        version: i64,
        checkpoint_type: CheckPointType,
        checkpoint_target: CheckPointTarget,
    ) -> Result<(bool, bool)> {
        if checkpoint_target == CheckPointTarget::CkpTargetNfs {
            Ok(self.finish_records.read().check_record(version))
        } else {
            Err(anyhow!(format!(
                "not supported yet! checkpoint_target: {}",
                checkpoint_target.as_str_name().clone()
            )))
        }
    }
}
