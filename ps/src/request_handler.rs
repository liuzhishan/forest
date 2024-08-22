use std::borrow::Borrow;
use std::cmp::min;
use std::iter::zip;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use grpc::sniper::heartbeat_option::{CheckStatus, StatusType};
use hashbrown::HashMap;
use log::{error, info};

use prost_types::{field_options, Any};
use tokio_graceful_shutdown::{SubsystemBuilder, Toplevel};
use tonic::{transport::Server, Code, Request, Response, Status};
use tonic_types::{ErrorDetails, StatusExt};

use grpc::sniper::sniper_ps_server::{SniperPs, SniperPsServer};
use grpc::sniper::{
    start_sample_option, CheckPointTarget, CheckPointType, CreateOption, DataType,
    EmbeddingLookupOption, FeedSampleOption, FreezeOption, GpuPsDenseData, HeartbeatOption,
    HelloRequest, HelloResponse, PullOption, PushGradOption, PushOption, RestoreOption, Role,
    SaveOption, StartSampleOption, TensorMessage, VoidMessage,
};

use grpc::sniper::{TensorProto, TensorShapeProto, VariableType};
use grpc::tool::{get_request_inner_options, send_bad_request_error, send_error_message};
use util::error_bail;

use crate::checkpoint::checkpoint_manager::CheckpointManager;
use crate::checkpoint::restore_task::{
    RestoreDenseFromLocalTask, RestoreSparseFromLocalTask, RestoreSparseTask,
};
use crate::checkpoint::save_task::{SaveDenseToLocalTask, SaveSparseTask, SaveSparseToLocalTask};
use crate::checkpoint::tool::CheckpointContext;
use crate::embedding::{Embedding, EmbeddingLookupResult};
use crate::env::Env;
use crate::scheduler::Scheduler;
use crate::variable_manager::{DenseManager, EmbeddingManager};

use dashmap::DashMap;
use dashmap::{mapref::one::Ref, RawRwLock};

pub type RwLock<T> = lock_api::RwLock<RawRwLock, T>;

/// Ps server.
///
/// Start ps for processing parameters.
pub struct Ps {
    /// Sparse embedding manager, add, remove, or update embedding parameters.
    embedding_manager: Arc<RwLock<EmbeddingManager>>,

    /// Dense parameter manager.
    dense_manager: Arc<RwLock<DenseManager>>,

    /// Checkpoint manager.
    checkpoint_manager: Arc<RwLock<CheckpointManager>>,

    /// Scheduer for tasks.
    scheduler: Arc<RwLock<Scheduler>>,

    /// Global Env for all request.
    env: Arc<RwLock<Env>>,
}

impl Ps {
    pub fn new() -> Self {
        let embedding_manager = Arc::new(RwLock::new(EmbeddingManager::default()));
        let dense_manager = Arc::new(RwLock::new(DenseManager::default()));
        let checkpoint_manager = Arc::new(RwLock::new(CheckpointManager::default()));
        let scheduler = Arc::new(RwLock::new(Scheduler::default()));
        let env = Arc::new(RwLock::new(Env::new()));

        Self {
            embedding_manager,
            dense_manager,
            checkpoint_manager,
            scheduler,
            env,
        }
    }

    /// Create embedding variable by create_option.
    fn create_embedding_variable(
        &self,
        varname: &String,
        create_option: &CreateOption,
    ) -> Result<()> {
        // Delete embedding variable for auto sharding.
        if create_option.r#type == VariableType::VarEmbedding.into() && create_option.delete_var {
            self.embedding_manager.write().remove(varname);
        }

        let env_read = self.env.read();

        let max_feed_queue_size = env_read.max_feed_queue_size;
        let max_lookup_queue_size = env_read.max_lookup_queue_size;

        self.embedding_manager.write().add_new_var(
            varname,
            create_option.emb_size as usize,
            create_option.shard_num as usize,
            create_option.shard_idx as usize,
            &create_option.fields,
            create_option.capacity as u64,
            create_option.hash_size as usize,
            max_feed_queue_size,
            max_lookup_queue_size,
        );

        Ok(())
    }

    /// Create dense variable by create_option.
    fn create_dense_variable(&self, varname: &String, create_option: &CreateOption) -> Result<()> {
        match create_option.shape.as_ref() {
            Some(shape) => {
                let dims = shape
                    .dim
                    .iter()
                    .map(|x| x.clone() as usize)
                    .collect::<Vec<_>>();

                self.dense_manager.write().add_new_var(varname, &dims);
                Ok(())
            }
            None => {
                error_bail!(
                    "missing shape in create_option! varname: {}",
                    varname.clone()
                );
            }
        }
    }

    /// Pull sparse parameters.
    fn pull_sparse(
        &self,
        varname: &String,
        batch_id: u64,
        pull_option: &PullOption,
    ) -> Result<TensorMessage> {
        let mut out_option = PullOption::default();

        let mut keys: Vec<u64> = Vec::new();
        let mut values: Vec<f32> = Vec::new();

        let embedding_manager_read = self.embedding_manager.read();
        let var_option = embedding_manager_read.get(varname);

        match var_option {
            Some(var) => {
                var.pull(
                    batch_id,
                    &pull_option,
                    &mut out_option,
                    &mut keys,
                    &mut values,
                );

                let res_options = match Any::from_msg(&out_option) {
                    Ok(x) => x,
                    Err(err) => {
                        error_bail!(
                            "encode Any from out_option failed! varname: {}, err: {}",
                            varname.clone(),
                            err,
                        );
                    }
                };

                let tensor1_dim = vec![keys.len() as i64];
                let tensor1 = TensorProto::with_vec(DataType::DtInt64.into(), &tensor1_dim, &keys);

                let tensor2_dim = vec![values.len() as i64];
                let tensor2 =
                    TensorProto::with_vec(DataType::DtFloat.into(), &tensor2_dim, &values);

                let response = TensorMessage {
                    role: Role::Ps.into(),
                    role_id: Into::<i32>::into(Role::Ps) as u32,
                    seq_id: batch_id.into(),
                    varname: varname.clone(),
                    options: Some(res_options),
                    tensor1: Some(tensor1),
                    tensor2: Some(tensor2),
                };

                return Ok(response);
            }
            None => {
                error_bail!("cannot find sparse var, varname: {}", varname.clone(),);
            }
        };
    }

    /// Pull dense parameters.
    fn pull_dense(
        &self,
        varname: &String,
        batch_id: u64,
        pull_option: &PullOption,
    ) -> Result<TensorMessage> {
        let out_option = PullOption::default();
        let mut values: Vec<f32> = Vec::new();

        let dense_manager_read = self.dense_manager.read();
        let var_option = dense_manager_read.get(varname);

        match var_option {
            Some(var) => {
                var.pull(&mut values);

                let res_options = match Any::from_msg(&out_option) {
                    Ok(x) => x,
                    Err(err) => {
                        error_bail!(
                            "encode any from out option failed! varname: {}, err: {}",
                            varname.clone(),
                            err,
                        );
                    }
                };

                let tensor1_dim = vec![values.len() as i64];
                let tensor1 =
                    TensorProto::with_vec(DataType::DtFloat.into(), &tensor1_dim, &values);

                let tensor2 = TensorProto::default();

                let response = TensorMessage {
                    role: Role::Ps.into(),
                    role_id: Into::<i32>::into(Role::Ps) as u32,
                    seq_id: batch_id.into(),
                    varname: varname.clone(),
                    options: Some(res_options),
                    tensor1: Some(tensor1),
                    tensor2: Some(tensor2),
                };

                return Ok(response);
            }
            None => {
                error_bail!("pull dense var failed, varname: {}", varname.clone(),);
            }
        }
    }

    /// Get embedding lookup result concurrently.
    pub async fn get_embedding_lookup_result(
        &self,
        varnames: &Vec<String>,
        batch_id: u64,
        lookup_option: &EmbeddingLookupOption,
    ) -> Result<Vec<EmbeddingLookupResult>> {
        let mut tasks = Vec::with_capacity(lookup_option.field_idx.len());

        for i in 0..varnames.len() {
            if i >= lookup_option.field_idx.len() {
                error_bail!(
                    "out of range when embedding lookup, i: {}, field_idx.len(): {}",
                    i,
                    lookup_option.field_idx.len(),
                );
            }

            let new_batch_id = batch_id;
            let new_varname = varnames[i].clone();
            let new_lookup_option = lookup_option.clone();
            let batch_size = lookup_option.batch_size as usize;

            let field = lookup_option.field_idx[i];

            let embedding_manager_clone = self.embedding_manager.clone();

            tasks.push(tokio::spawn(async move {
                let embedding_manager_read = embedding_manager_clone.read();
                let var_option = embedding_manager_read.get(&new_varname);

                match var_option {
                    Some(var) => var.embedding_lookup(field, new_batch_id, batch_size),
                    None => Err(anyhow!(format!(
                        "cannot find embedding by varname: {}",
                        new_varname.clone()
                    ))),
                }
            }));
        }

        let mut lookup_res: Vec<EmbeddingLookupResult> = Vec::with_capacity(varnames.len());
        let mut error_messages: Vec<String> = Vec::new();

        for task in tasks {
            match task.await {
                Ok(task_res) => match task_res {
                    Ok(x) => {
                        lookup_res.push(x);
                    }
                    Err(err) => {
                        error_messages.push(err.to_string());
                    }
                },
                Err(err) => {
                    error_messages.push(err.to_string());
                }
            }
        }

        if error_messages.len() == 0 {
            Ok(lookup_res)
        } else {
            Err(anyhow!(error_messages.join(", ")))
        }
    }

    /// Update grad parameters.
    async fn update_grad_parameters(
        &self,
        varnames: &Vec<String>,
        batch_id: u64,
        batch_size: usize,
        values: &[f32],
        option: &PushGradOption,
    ) -> Result<()> {
        let mut tasks = Vec::with_capacity(option.field_idx.len());

        // var_start_index is used to get the index of float values for each varname and field.
        let mut var_start_index: Vec<usize> = Vec::with_capacity(option.field_idx.len());
        var_start_index.resize(option.field_idx.len(), 0);

        for (i, dim) in option.field_dim.iter().enumerate() {
            if i < option.field_dim.len() - 1 {
                var_start_index[i + 1] = var_start_index[i] + *dim as usize;
            }
        }

        if varnames.len() != option.field_idx.len() {
            error_bail!(
                "varnames.len() != option.field_idx.len(), varnames.len(): {}, field_idx.len(): {}",
                varnames.len(),
                option.field_idx.len(),
            );
        }

        if varnames.len() != option.field_dim.len() {
            error_bail!(
                "varnames.len() != option.field_dim.len(), varnames.len(): {}, field_dim.len(): {}",
                varnames.len(),
                option.field_dim.len(),
            );
        }

        let total = varnames.len();

        for i in 0..total {
            let varname = varnames[i].clone();

            if i >= option.field_idx.len() {
                error_bail!(
                    "out of range when push grad, varname: {}, i: {}, field_idx.len(): {}",
                    varname.clone(),
                    i,
                    option.field_idx.len(),
                );
            }
            let field = option.field_idx[i].clone();

            let new_batch_id = batch_id;
            let new_option = option.clone();
            let batch_size = option.batch_size as usize;

            let learning_rate = option.learning_rate;
            let eps = option.eps;
            let eta = option.eta;
            let decay = option.decay;
            let l2 = option.l2;

            // Only use the float values in corresponding index.
            let start_index = var_start_index[i];
            let end_index = start_index + option.field_dim[i] as usize;

            let grad_values = values[start_index..end_index].to_vec();

            let embedding_manager_clone = self.embedding_manager.clone();

            tasks.push(tokio::spawn(async move {
                let embedding_manager_write = embedding_manager_clone.write();
                let var_option = embedding_manager_write.get(&varname);

                match var_option {
                    Some(var) => var.push_grad(
                        grad_values.as_slice(),
                        new_batch_id,
                        field,
                        learning_rate,
                        eta,
                        eps,
                        decay,
                        l2,
                    ),
                    None => Err(anyhow!(format!(
                        "cannot find embedding by varname: {}",
                        varname.clone()
                    ))),
                }
            }));
        }

        let mut error_messages: Vec<String> = Vec::new();

        for task in tasks {
            match task.await {
                Ok(task_res) => match task_res {
                    Ok(_) => {}
                    Err(err) => {
                        error_messages.push(err.to_string());
                    }
                },
                Err(err) => {
                    error_messages.push(err.to_string());
                }
            }
        }

        if error_messages.len() == 0 {
            Ok(())
        } else {
            Err(anyhow!(error_messages.join(", ")))
        }
    }

    /// Compute total bucket number for total, each bucket has max `max_save_key_size_in_file` items.
    #[inline]
    fn get_bucket_number(&self, total: u64, max_save_key_size_in_file: u64) -> u64 {
        if total % max_save_key_size_in_file == 0 {
            total / max_save_key_size_in_file
        } else {
            total / max_save_key_size_in_file + 1
        }
    }

    /// Save sparse embedding.
    ///
    /// Create new async task and dispatch the task, then running the task in background. Other request will
    /// check whether the save is done.
    async fn save_sparse_variable(&self, varname: &String, save_option: &SaveOption) -> Result<()> {
        // Split the parameters into groups, each group save max `env.max_save_key_size_in_file` signs. Each
        // task is assigned a `inner_shard` and `inner_shard_total`. We use `sign % inner_shard_total == inner_shard`
        // to determine whether the sign should be handled by the task. Thus every sign is assigned to exactly only
        // one task,
        let embedding_manager_read = self.embedding_manager.read();
        let var_option = embedding_manager_read.get(varname);

        let max_save_key_size_in_file = self.env.read().max_save_key_size_in_file;

        match var_option {
            Some(var) => {
                let embedding = var.value();

                let total = embedding.store.len() as u64;
                let bucket_number =
                    self.get_bucket_number(total, max_save_key_size_in_file) as usize;

                let mut context = CheckpointContext::default();

                context.version = 1;
                context.checkpoint_type = CheckPointType::CkpTypeFull.into();

                // Only support hdfs and local now.
                context.checkpoint_target = CheckPointTarget::CkpTargetNfs.into();

                context.path = save_option.nfs_path.clone();
                context.model_name = self.env.read().model_name.clone();
                context.varname = varname.clone();
                context.variable_type = save_option.variable_type();
                context.shard_index = embedding.shard_index as i32;
                context.shard_num = embedding.shard_num as i32;
                context.need_finished = true;
                context.has_finished = false;
                context.inner_shard_total = bucket_number;
                context.max_record_iterate_count = 200_000;
                context.variable_dim = embedding.embedding_size;

                // Optimizer parameter dim is same as embedding size for adagrad. Right now only adagrad is supported, so the
                // value is fixed.
                //
                // TODO: More Optimizer support, optimizer_dim should be consistent with different optimizer.
                context.optimizer_dim = embedding.embedding_size;

                for i in 0..bucket_number {
                    let mut new_context = context.clone();

                    new_context.inner_shard = i as i32;

                    let embedding_manager_clone = self.embedding_manager.clone();

                    // Dispatch the save task.
                    tokio::spawn(async move {
                        let save_task = SaveSparseToLocalTask::new(&new_context);

                        let embedding_manager_read = embedding_manager_clone.read();
                        let var_option = embedding_manager_read.get(&new_context.varname);

                        match var_option {
                            Some(var) => match save_task.run(var.value()) {
                                Ok(_) => {
                                    info!(
                                            "save one shard of embeding done, varname: {}, inner_shard: {}",
                                            new_context.varname.clone(),
                                            new_context.inner_shard,
                                        );

                                    Ok(())
                                }
                                Err(err) => {
                                    error_bail!(
                                            "save embedding parameters failed! varname: {}, inner_shard: {}, err: {}",
                                            new_context.varname.clone(),
                                            new_context.inner_shard,
                                            err,
                                        );
                                }
                            },
                            None => {
                                error_bail!(
                                    "cannot find var in embedding_manager when save, varname: {}",
                                    new_context.varname.clone(),
                                );
                            }
                        }
                    });
                }

                Ok(())
            }
            None => {
                error_bail!(
                    "cannot find var in embedding_manager, varname: {}",
                    varname.clone(),
                );
            }
        }
    }

    /// Save dense variable parameters.
    async fn save_dense_variable(&self, varname: &String, save_option: &SaveOption) -> Result<()> {
        let dense_manager_read = self.dense_manager.read();
        let var_option = dense_manager_read.get(varname);

        match var_option {
            Some(var) => {
                let dense = var.value();

                let mut context = CheckpointContext::default();

                context.version = 1;
                context.checkpoint_type = CheckPointType::CkpTypeFull.into();

                // Only support hdfs and local now.
                context.checkpoint_target = CheckPointTarget::CkpTargetNfs.into();

                context.path = save_option.nfs_path.clone();
                context.model_name = self.env.read().model_name.clone();
                context.varname = varname.clone();
                context.variable_type = save_option.variable_type();
                context.need_finished = true;
                context.has_finished = false;
                context.variable_dim = dense.values.len();

                let dense_manager_clone = self.dense_manager.clone();

                tokio::spawn(async move {
                    let save_task = SaveDenseToLocalTask::new(&context);

                    let dense_manager_read = dense_manager_clone.read();
                    let var_option = dense_manager_read.get(&context.varname);

                    match var_option {
                        Some(var) => match save_task.run(var.value()) {
                            Ok(_) => {
                                info!(
                                    "save dense to file success! varname: {}",
                                    context.varname.clone()
                                );

                                Ok(())
                            }
                            Err(err) => {
                                error_bail!(
                                    "save dense to file failed! varname: {}",
                                    context.varname.clone(),
                                );
                            }
                        },
                        None => {
                            error_bail!(
                                "cannot find var in dense_manager, varname: {}",
                                context.varname.clone(),
                            );
                        }
                    }
                });
            }
            None => {
                error_bail!(
                    "cannot find var in dense_manager, varname: {}",
                    varname.clone(),
                );
            }
        }

        Ok(())
    }

    /// Restore sparse embedding parameters from file.
    ///
    /// Dispatch RestoreTask by parameters.
    async fn restore_sparse_variable(
        &self,
        varname: &String,
        restore_option: &RestoreOption,
    ) -> Result<()> {
        let weight_paths = &restore_option.nfs_weight_path;
        let adagrad_paths = &restore_option.nfs_adagrad_path;

        if weight_paths.len() != adagrad_paths.len() {
            error_bail!(
                "weight_paths.len() != adagrad_paths.len(), weight_paths.len(): {}, adagra_paths.len(): {}",
                weight_paths.len(),
                adagrad_paths.len(),
            );
        }

        let embedding_manager_read = self.embedding_manager.read();
        let var_option = embedding_manager_read.get(varname);

        match var_option {
            Some(var) => {
                // Loop over paths, for each weight_path and adagrad_path, spawn a new restore task.
                for (weight_path, adagrad_path) in zip(weight_paths, adagrad_paths) {
                    let mut context = CheckpointContext::default();

                    context.checkpoint_target = CheckPointTarget::CkpTargetNfs;
                    context.model_name = self.env.read().model_name.clone();
                    context.path = format!("{},{}", weight_path.clone(), adagrad_path.clone());
                    context.varname = varname.clone();

                    context.shard_index = var.value().shard_index as i32;
                    context.shard_num = var.value().shard_num as i32;

                    let embedding_manager_clone = self.embedding_manager.clone();

                    tokio::spawn(async move {
                        let embedding_manager_write = embedding_manager_clone.write();
                        let embedding_var_option =
                            embedding_manager_write.get_mut(&context.varname);

                        match embedding_var_option {
                            Some(embedding_var) => {
                                let restore_task = RestoreSparseFromLocalTask::new(&context);

                                match restore_task.run(embedding_var.value()) {
                                    Ok(_) => {
                                        info!(
                                            "restore embedding success! varname: {}, shard_index: {}, filename: {}",
                                            context.varname.clone(),
                                            context.shard_index,
                                            context.path.clone(),
                                        );

                                        Ok(())
                                    }
                                    Err(err) => {
                                        error_bail!(
                                            "restore embedding failed! varname: {}, shard_index: {}, filename: {}",
                                            context.varname.clone(),
                                            context.shard_index,
                                            context.path.clone(),
                                        );
                                    }
                                }
                            }
                            None => {
                                error_bail!(
                                    "cannot find var in embedding_manager, varname: {}",
                                    context.varname.clone(),
                                );
                            }
                        }
                    });
                }

                Ok(())
            }
            None => {
                error_bail!(
                    "cannot find var in embedding_manager, varname: {}",
                    varname.clone(),
                );
            }
        }
    }

    /// Restore dense embedding parameters from file.
    ///
    /// Dispatch RestoreTask by parameters.
    async fn restore_dense_variable(
        &self,
        varname: &String,
        restore_option: &RestoreOption,
    ) -> Result<()> {
        let model_name = self.env.read().model_name.clone();

        let weight_paths = &restore_option.nfs_weight_path;

        // Dense variable has only one path.
        if weight_paths.len() != 1 {
            error_bail!(
                "weight_paths of dense variable must be 1, but is weight_paths.len(): {}, varname: {}, paths: {}",
                weight_paths.len(),
                varname.clone(),
                weight_paths.join(", "),
            );
        }

        let mut context = CheckpointContext::default();

        context.checkpoint_target = CheckPointTarget::CkpTargetNfs;
        context.model_name = model_name.clone();
        context.path = weight_paths[0].clone();
        context.varname = varname.clone();

        let dense_manager_clone = self.dense_manager.clone();

        tokio::spawn(async move {
            let restore_task = RestoreDenseFromLocalTask::new(&context);

            let dense_manager_write = dense_manager_clone.write();
            let dense_var_option = dense_manager_write.get_mut(&context.varname);

            match dense_var_option {
                Some(mut dense_var) => match restore_task.run(&mut dense_var.value_mut()) {
                    Ok(_) => {
                        info!(
                            "restore dense var success! varname: {}, path: {}",
                            context.varname.clone(),
                            context.path.clone(),
                        );

                        Ok(())
                    }
                    Err(err) => {
                        error_bail!(
                            "restore dense var failed! varname: {}, path: {}, err: {}",
                            context.varname.clone(),
                            context.path.clone(),
                            err,
                        );
                    }
                },
                None => {
                    error_bail!(
                        "cannot find var in dense_manager, varname: {}",
                        context.varname.clone(),
                    );
                }
            }
        });

        Ok(())
    }
}

#[tonic::async_trait]
impl SniperPs for Ps {
    /// Create ps vars embedding table and dense var parameters in ps.
    async fn create(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();
        let varname = &request_inner.varname;

        // Get create_option from request.options.
        let create_option: CreateOption =
            match get_request_inner_options::<CreateOption>(&request_inner) {
                Some(x) => x,
                None => {
                    return send_bad_request_error("options", "options is invalid CreateOption.");
                }
            };

        if create_option.r#type == VariableType::VarEmbedding.into() {
            match self.create_embedding_variable(varname, &create_option) {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "create embedding variable failed! varname: {}, error: {}",
                    varname.clone(),
                    err
                )),
            }
        } else {
            match self.create_dense_variable(varname, &create_option) {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "create dense variable failed! varname: {}, err: {}",
                    varname.clone(),
                    err
                )),
            }
        }
    }

    /// Add schedule of sparse queue and dense queues, set Env default value.
    ///
    /// Queue is used for futher increment parameter sending, right now all queue is empty. Just leave
    /// the interface for more extensiblity in the future.
    async fn freeze(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();
        let freeze_option = match get_request_inner_options::<FreezeOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "options is invalid FreezeOption.");
            }
        };

        let hub_endpoints = freeze_option.hub_eps.clone();
        let ps_endpoints = freeze_option.ps_eps.clone();

        let dense_vars: HashMap<String, String> = HashMap::default();
        let sparse_vars: HashMap<String, String> = HashMap::default();

        let mut ps_shard: HashMap<String, Vec<String>> = HashMap::default();

        freeze_option.ps_shard.iter().for_each(|x| {
            ps_shard.insert(x.0.clone(), x.1.value.clone());
        });

        // self.env.assemble(&sparse_vars);

        if !self.scheduler.write().init(
            &freeze_option.scheduler_ep,
            &hub_endpoints,
            &ps_endpoints,
            &dense_vars,
            &sparse_vars,
            &ps_shard,
        ) {
            return send_error_message::<VoidMessage>("scheduler init failed!");
        }

        if freeze_option.is_scheduler {
            let mut env_write = self.env.write();

            env_write.role = Role::Scheduler.into();

            // restart schduler.
            let mut scheduler = self.scheduler.write();

            scheduler.stop();
            scheduler.start();
        }

        Ok(Response::new(VoidMessage::default()))
    }

    /// FeedSample.
    ///
    /// Get EmbeddingTable from EmbeddingManager, iterator field in field_infos, and put data
    /// into Embedding.
    async fn feed_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        let varname = request_inner.varname.clone();
        let batch_id = request_inner.seq_id;

        let feed_sample_option = match get_request_inner_options::<FeedSampleOption>(&request_inner)
        {
            Some(x) => x,
            None => {
                return send_bad_request_error(
                    "options",
                    "cannot get feed_sample_option from request!",
                );
            }
        };

        match self.embedding_manager.read().get(&varname) {
            Some(embedding) => {
                let batch_size = feed_sample_option.batch_size;

                let field_infos = &feed_sample_option.field_info;

                for i in 0..field_infos.len() {
                    match embedding.feed_sample(batch_id, &feed_sample_option, i) {
                        Ok(_) => {}
                        Err(err) => {
                            return send_error_message::<VoidMessage>(format!(
                                "feed sample failed! varname: {}, err: {}",
                                varname.clone(),
                                err,
                            ));
                        }
                    }
                }
            }
            None => {
                return send_error_message(format!(
                    "cannot find varname in embedding_manager, varname: {}",
                    varname.clone()
                ));
            }
        }

        Ok(Response::new(VoidMessage::default()))
    }

    /// Push.
    ///
    /// Push variable parameters to manager. Use push_option.variable_type to distinguish sparse
    /// or dense.
    async fn push(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        let batch_id = request_inner.seq_id;
        let varname = request_inner.varname.clone();

        let push_option = match get_request_inner_options::<PushOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "cannot get push_option in request!");
            }
        };

        let keys: &[u64] = match request_inner.tensor1.as_ref() {
            Some(x) => x.as_slice::<u64>(),
            None => {
                return send_bad_request_error::<VoidMessage>(
                    "tensor1",
                    format!("tensor1 is None! varname: {}", varname.clone()),
                );
            }
        };

        let values: &[f32] = match request_inner.tensor1.as_ref() {
            Some(x) => x.as_slice::<f32>(),
            None => {
                return send_bad_request_error::<VoidMessage>(
                    "tensor2",
                    format!("tensor2 is None! varname: {}", varname.clone()),
                );
            }
        };

        if push_option.variable_type == VariableType::VarEmbedding.into() {
            match self.embedding_manager.write().get(&varname) {
                Some(var) => match var.value().push(batch_id, &push_option, keys, values) {
                    Ok(_) => {}
                    Err(err) => {
                        return send_error_message(format!(
                            "push embedding var failed! varname: {}, err: {}",
                            varname.clone(),
                            err,
                        ));
                    }
                },
                None => {
                    return send_error_message(format!(
                        "cannot find embedding variable, varname: {}",
                        varname.clone()
                    ));
                }
            }
        } else {
            match self.dense_manager.write().get_mut(&varname) {
                Some(mut var) => match var.push(values) {
                    Ok(_) => {}
                    Err(err) => {
                        return send_error_message(format!(
                            "push dense var failed! varname: {}, err: {}",
                            varname.clone(),
                            err,
                        ));
                    }
                },
                None => {
                    return send_error_message(format!(
                        "cannot find dense variable, varname: {}",
                        varname.clone()
                    ));
                }
            }
        }

        Ok(Response::new(VoidMessage::default()))
    }

    /// Pull the parameters by varname.
    async fn pull(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        let request_inner = request.into_inner();

        let batch_id = request_inner.seq_id;
        let varname = request_inner.varname.clone();

        let pull_option = match get_request_inner_options::<PullOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "cannot get pull_option in request!");
            }
        };

        if pull_option.variable_type == VariableType::VarEmbedding.into() {
            match self.pull_sparse(&varname, batch_id, &pull_option) {
                Ok(response) => {
                    return Ok(Response::new(response));
                }
                Err(err) => {
                    return send_error_message::<TensorMessage>(format!(
                        "pull sparse failed, varname: {},  err: {}",
                        varname.clone(),
                        err,
                    ));
                }
            }
        } else {
            match self.pull_dense(&varname, batch_id, &pull_option) {
                Ok(response) => {
                    return Ok(Response::new(response));
                }
                Err(err) => {
                    return send_error_message::<TensorMessage>(format!(
                        "pull dense var failed, varname: {}, err; {}",
                        varname.clone(),
                        err,
                    ));
                }
            }
        }
    }

    /// EmbeddingLookup.
    ///
    /// Get embedding lookup result for multiple varnames. Get the result of lookup for each varname
    /// concurrently, then assembly the result to 2d tensor, whose first dimension is batch_size, second
    /// dimension is the sum of all var embedding size.
    ///
    /// Example:
    /// 2 varnames, batch_size: 4, embedding_size: 2
    ///
    /// [[1.1, 3.3, 4.4, 6.6],
    ///  [0.9, 0.7, 0.4, 0.3],
    ///  [0.5, 0.7, 0.1, 0.3],
    ///  [0.4, 0.2, 1.1, 9.9]]
    async fn embedding_lookup(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        let request_inner = request.into_inner();

        let batch_id = request_inner.seq_id;

        let varnames = request_inner
            .varname
            .split(",")
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        if varnames.len() == 0 {
            return send_bad_request_error("options", "varname is empty!");
        }

        let lookup_option = match get_request_inner_options::<EmbeddingLookupOption>(&request_inner)
        {
            Some(x) => x,
            None => {
                return send_bad_request_error(
                    "options",
                    "options is invalid EmbeddingLookupOption!",
                );
            }
        };

        // Get embedding lookup result concurrently.
        //
        // Each varname in varnames is exactly coresponding to one lookup_option.field_idx. With the
        // same index.
        if varnames.len() != lookup_option.field_idx.len() {
            return send_bad_request_error(
                "options",
                format!(
                    "varnames.len() != field_idx.len()! varnames.len(): {}, field_idx.len(): {}",
                    varnames.len(),
                    lookup_option.field_idx.len()
                ),
            );
        }

        let lookup_res = match self
            .get_embedding_lookup_result(&varnames, batch_id, &lookup_option)
            .await
        {
            Ok(x) => x,
            Err(err) => {
                return send_error_message::<TensorMessage>(format!(
                    "embedding lookup failed! varnames: {}, error: {}",
                    varnames.join(","),
                    err,
                ));
            }
        };

        // Assembly the result.
        let batch_size = lookup_option.batch_size as usize;

        // Compute the total embedding size of all field.
        let dim_acc: usize = lookup_option
            .field_dim
            .iter()
            .fold(0, |acc, x| acc + *x as usize);

        let res_dim_acc: usize = lookup_res
            .iter()
            .filter(|x| x.values.len() > 0)
            .fold(0, |acc, x| acc + x.values[0].len());

        if dim_acc != res_dim_acc {
            return send_error_message::<TensorMessage>(format!(
                "dim is not eqaul! lookup res total dim: {}, option total dim: {}",
                res_dim_acc, dim_acc,
            ));
        }

        let total = lookup_option.batch_size as usize * dim_acc;

        let mut parameters: Vec<f32> = Vec::with_capacity(total);

        // Assembly the result to 2d tensor.
        // First dimension is batch_size, second dimension is field.
        for i in 0..batch_size {
            for j in 0..lookup_res.len() {
                if i < lookup_res[j].values.len() {
                    parameters.extend_from_slice(lookup_res[j].values[i].as_slice());
                } else {
                    return send_error_message::<TensorMessage>(format!(
                        "out of range, i: {}, j: {}, lookup_res[j].values.len(): {}",
                        i,
                        j,
                        lookup_res[j].values.len(),
                    ));
                }
            }
        }

        let dim = vec![total as i64];
        let tensor1 = TensorProto::with_vec(DataType::DtFloat.into(), &dim, &parameters);

        let response = TensorMessage {
            role: Role::Ps.into(),
            role_id: Into::<i32>::into(Role::Ps) as u32,
            seq_id: batch_id.into(),
            varname: request_inner.varname.clone(),
            options: None,
            tensor1: Some(tensor1),
            tensor2: None,
        };

        Ok(Response::new(response))
    }

    /// PushGrad.
    async fn push_grad(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        let batch_id = request_inner.seq_id;

        let varnames = request_inner
            .varname
            .split(",")
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        if varnames.len() == 0 {
            return send_bad_request_error("options", "varname is empty!");
        }

        let push_grad_option = match get_request_inner_options::<PushGradOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error::<VoidMessage>(
                    "options",
                    "options is invalid PushGradOption!",
                );
            }
        };

        // Get the grad parameters from tensor1.
        let values: &[f32] = match request_inner.tensor1.as_ref() {
            Some(x) => x.as_slice::<f32>(),
            None => {
                return send_bad_request_error::<VoidMessage>("tensor1", "tensor1 is None!");
            }
        };

        let batch_size = push_grad_option.batch_size as usize;

        match self
            .update_grad_parameters(&varnames, batch_id, batch_size, values, &push_grad_option)
            .await
        {
            Ok(_) => Ok(Response::new(VoidMessage::default())),
            Err(err) => {
                return send_error_message::<VoidMessage>(format!(
                    "push grad failed! varnames: {}, batch_id: {}",
                    request_inner.varname.clone(),
                    batch_id,
                ));
            }
        }
    }

    /// Save.
    ///
    /// Save embedding parameter to hdfs. Each sparse variable is save to multiple files.
    async fn save(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        let varname = &request_inner.varname;

        let save_option = match get_request_inner_options::<SaveOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "options is invalid SaveOption!");
            }
        };

        if save_option.variable_type == VariableType::VarEmbedding.into() {
            match self.save_sparse_variable(&varname, &save_option).await {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "save sparse variable failed! varname: {}, err: {}",
                    varname.clone(),
                    err,
                )),
            }
        } else {
            match self.save_dense_variable(&varname, &save_option).await {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "save dense variable failed! varname: {}, err: {}",
                    varname.clone(),
                    err,
                )),
            }
        }
    }

    /// Restore.
    ///
    /// Restore parameters from path.
    async fn restore(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        let varname = &request_inner.varname;

        let restore_option = match get_request_inner_options::<RestoreOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "options is invalid RestoreOption!");
            }
        };

        if restore_option.variable_type == VariableType::VarEmbedding.into() {
            match self
                .restore_sparse_variable(&varname, &restore_option)
                .await
            {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "restore sparse variable failed! varname: {}, err: {}",
                    varname.clone(),
                    err,
                )),
            }
        } else {
            match self.restore_dense_variable(&varname, &restore_option).await {
                Ok(_) => Ok(Response::new(VoidMessage::default())),
                Err(err) => send_error_message::<VoidMessage>(format!(
                    "restore dense variable failed! varname: {}, err: {}",
                    varname.clone(),
                    err,
                )),
            }
        }
    }

    /// Complete.
    async fn complete(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // Nothing to do.
        Ok(Response::new(VoidMessage::default()))
    }

    /// Heartbeat.
    ///
    /// The server is on the only scheduler ps, the client is all ps.
    /// The scheduler maintains infomation of all checkpoint task status. These statuses can be used to determin
    /// where save or restore is done.
    ///
    /// Response with following infomation.
    /// 1. Update stat info of one checkpoint task after task is done on scheduler ps. The request is sent after
    /// save task done from worker ps.
    /// 2. Whether checkpoint task is finished and success.
    async fn heartbeat(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        let request_inner = request.into_inner();

        let heartbeat_option = match get_request_inner_options::<HeartbeatOption>(&request_inner) {
            Some(x) => x,
            None => {
                return send_bad_request_error("options", "options is not valid HeartbeatOption!");
            }
        };

        let st_type = heartbeat_option.st_type();

        if st_type == StatusType::TypeCkp {
            // Update checkpoint task status.
            match heartbeat_option.ckp_st.as_ref() {
                Some(checkpoint_status) => {
                    match self.scheduler.write().post_checkpoint_status(
                        checkpoint_status.incr,
                        checkpoint_status.version,
                        &checkpoint_status.var_name,
                        checkpoint_status.success,
                        checkpoint_status.ckp_target(),
                        &checkpoint_status.nfs_path,
                        checkpoint_status.shard_idx,
                        checkpoint_status.shard_num,
                    ) {
                        Ok(_) => Ok(Response::new(TensorMessage::default())),
                        Err(err) => {
                            return send_error_message::<TensorMessage>(format!(
                                "post_checkpoint_status failed! varname: {}, err: {}",
                                checkpoint_status.var_name.clone(),
                                err,
                            ));
                        }
                    }
                }
                None => {
                    return send_bad_request_error("HeartbeatOption", "options has no ckp_st!");
                }
            }
        } else {
            // Decide whether all task is finished and success.
            // Request must provide check_st in options.
            let check_st = match heartbeat_option.check_st.as_ref() {
                Some(x) => x,
                None => {
                    return send_bad_request_error("check_st", "check_st is None!");
                }
            };

            match self.scheduler.write().check_checkpoint_status(
                check_st.version,
                check_st.ckp_type(),
                check_st.ckp_target(),
            ) {
                Ok((is_finished, is_success)) => {
                    let mut res_st = CheckStatus::default();

                    res_st.finish = is_finished;
                    res_st.succ = is_success;

                    let mut out_option = HeartbeatOption::default();
                    out_option.check_st = Some(res_st);

                    let res_options = match Any::from_msg(&out_option) {
                        Ok(x) => x,
                        Err(err) => {
                            return send_error_message::<TensorMessage>(format!(
                                "out_option to Any failed! err: {}",
                                err
                            ));
                        }
                    };

                    let mut response = TensorMessage::default();
                    response.options = Some(res_options);

                    Ok(Response::new(response))
                }
                Err(err) => {
                    return send_error_message::<TensorMessage>(format!(
                        "check checkpoint status failed! err: {}",
                        err
                    ));
                }
            }
        }
    }
}
