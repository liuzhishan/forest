use anyhow::{anyhow, bail, Result};
use grpc::sniper::{FeedFieldInfo, GpuPsFeature64, Role, StartSampleOption, TensorMessage};
use log::{error, info};

use hashbrown::HashMap;

use prost::Message;
use prost_types::Any;
use ps::get_ps_client;
use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};
use util::{error_bail, get_target_shard_by_sign, FeaturePlacement};

use super::sample::SampleBatch;
use grpc::sniper::TensorProto;
use grpc::sniper::{sniper_client::SniperClient, FeedSampleOption};

/// Send features to ps and trainer.
///
/// For each SampleBatch, first generate a random u64 as batch_id, then send batch_id and sparse
/// features to ps, and send batch_id, dense_features and labels to trainer.
///
/// For each sparse field, we need to decide which ps to send. It's determined by FeaturePlament.
pub struct FeedSample {
    /// StartSampleOption.
    option: StartSampleOption,

    /// Receiver from BatchAssembler or BatchReader.
    sample_batch_receiver: async_channel::Receiver<SampleBatch>,

    /// FeaturePlacement is used to determine which ps each var live in.
    feature_placement: FeaturePlacement,

    /// Total batch.
    total_batch: i64,

    /// Send batch_id, dense features, labels to trainer.
    trainer_data_sender: async_channel::Sender<SampleBatch>,

    /// All ps_endpoints. Will be used for FeedSample.
    ps_endpoints: Vec<String>,

    /// All ps clients. The key is ps_endpoint.
    ps_clients: HashMap<String, SniperClient<tonic::transport::Channel>>,
}

impl FeedSample {
    pub fn new(
        option: StartSampleOption,
        sample_batch_receiver: async_channel::Receiver<SampleBatch>,
        feature_placement: FeaturePlacement,
        trainer_data_sender: async_channel::Sender<SampleBatch>,
        ps_endpoints: &Vec<String>,
    ) -> Self {
        Self {
            option,
            sample_batch_receiver,
            feature_placement,
            total_batch: 0,
            trainer_data_sender,
            ps_endpoints: ps_endpoints.clone(),
            ps_clients: HashMap::new(),
        }
    }

    /// Initialize.
    pub async fn init(&mut self) -> bool {
        // Try to get all ps clients.
        for ps_endpoint in self.ps_endpoints.iter() {
            let ps_name = ps_endpoint.clone();

            let ps_client = match get_ps_client(ps_name.clone()).await {
                Ok(client) => client,
                Err(err) => {
                    error!(
                        "get_ps_client in hub failed! ps_endpoint: {}, err: {}",
                        ps_name.clone(),
                        err
                    );
                    return false;
                }
            };

            self.ps_clients.insert(ps_endpoint.clone(), ps_client);
        }

        true
    }

    /// Send batch_id and sparse features to ps.
    pub async fn send_to_ps(&mut self, sample_batch: &SampleBatch) -> Result<()> {
        let sparse_feature_count = sample_batch.sparse_signs.len();

        // For extensability, the value of sparse_features is a HashMap of String
        // and FeedSampleOption. Each key of the inner map is ps_endpoint. Right now
        // there are only one ps_endpoint for each sparse embedding table.
        //
        // EmbeddingTable -> ps_endpoint -> FeedSamleOption
        let mut sparse_features: HashMap<String, HashMap<String, FeedSampleOption>> =
            HashMap::new();

        if self.option.feature_list.is_none() {
            error_bail!("option.feature_list is none!");
        }

        let feature_list = self.option.feature_list.as_ref().unwrap();

        // Must copy to avoid immutable and mutable borrow of self at the same time.
        let field_dims = feature_list.sparse_field_count.clone();
        let sparse_emb_table = feature_list.sparse_emb_table.clone();

        // Iterator all sparse feature signs, get the corresponding ps address.
        for (field, signs) in sample_batch.sparse_signs.iter().enumerate() {
            let field_dim = field_dims[field] as i32;

            // Get all data send to ps.
            // ps_endpoint -> FeedSampleOption
            let mut ps_features: HashMap<String, FeedSampleOption> = HashMap::new();

            // Assembly FeedSampleOption.
            let mut feed_sample_option = FeedSampleOption::default();
            feed_sample_option.batch_size = sample_batch.batch_size as u32;
            feed_sample_option.work_mode = self.option.work_mode;

            // Assembly field_info.
            let mut field_info = FeedFieldInfo::default();

            if field >= sample_batch.item_indexes.len() {
                error_bail!(
                    "out of range, field: {}, sample_batch.item_indexes.len(): {}",
                    field,
                    sample_batch.item_indexes.len(),
                );
            }

            let item_indexes = &sample_batch.item_indexes[field];

            field_info.field_idx = field as i32;
            field_info.field_dim = field_dim;

            if field >= sparse_emb_table.len() {
                error_bail!(
                    "out of range, field: {}, feature_list.sparse_emb_table.len(): {}",
                    field,
                    sparse_emb_table.len(),
                );
            }

            let varname = &sparse_emb_table[field];

            let feature_shards = match self.get_feature_shards(signs, item_indexes, varname) {
                Ok(x) => x,
                Err(err) => {
                    error_bail!("get_feature_shards failed! err: {}", err);
                }
            };

            feature_shards.into_iter().for_each(|(k, v)| {
                let mut new_field_info = field_info.clone();
                new_field_info.feature = v;

                let mut new_feed_sample_option = feed_sample_option.clone();
                new_feed_sample_option.field_info.push(new_field_info);

                ps_features.insert(k, new_feed_sample_option);
            });

            // insert varname and signs to sparse_features
            sparse_features.insert(varname.clone(), ps_features);
        }

        // Send the signs to ps. Wait for the response too check whether it's success.
        let mut handles = Vec::new();

        for (varname, inner) in sparse_features.iter() {
            for (ps_endpoint, feed_sample_option) in inner.iter() {
                let new_varname = varname.clone();
                let options = Any::from_msg(feed_sample_option)?;

                let batch_id = sample_batch.batch_id;

                // Get ps client by ps_endpoint.
                let mut client = match self.ps_clients.get(ps_endpoint) {
                    Some(client) => client.clone(),
                    None => {
                        error_bail!(
                            "cannot find ps_client, ps_endpoint: {}",
                            ps_endpoint.clone(),
                        );
                    }
                };

                handles.push(tokio::spawn(async move {
                    // TODO: use real seq_id,
                    let tensor_message = TensorMessage {
                        role: Role::Hub.into(),
                        role_id: Into::<i32>::into(Role::Hub) as u32,
                        seq_id: batch_id,
                        varname: new_varname,
                        options: Some(options),
                        tensor1: Some(TensorProto::default()),
                        tensor2: Some(TensorProto::default()),
                    };

                    let request = tonic::Request::new(tensor_message);

                    client.feed_sample(request).await
                }));
            }
        }

        for handle in handles {
            match handle.await {
                Ok(_) => {}
                Err(err) => {
                    error!("Hub feed sample failed! error: {}", err);
                }
            }
        }

        Ok(())
    }

    /// Split sparse features into different ps based on sign and shard_num.
    ///
    /// The key of result is ps_endpoint, the value of result is string of GpuPsFeature64 of one ps shard.
    fn get_feature_shards(
        &mut self,
        signs: &Vec<u64>,
        item_indexes: &Vec<usize>,
        embedding_varname: &String,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let mut res: HashMap<String, Vec<u8>> = HashMap::new();

        if signs.len() != item_indexes.len() {
            error_bail!(
                "signs.len() is not equal to item_indexes.len()! signs.len(): {}, item_indexes.len(): {}",
                signs.len(),
                item_indexes.len()
            );
        }

        let total_signs = signs.len();

        // Get ps shard.
        let shard_endpoints_opt = self.feature_placement.get_emb_placement(embedding_varname);
        if shard_endpoints_opt.is_none() {
            error_bail!(
                "cannot get shard_endpoints for varname: {}",
                embedding_varname.clone()
            );
        }

        let shard_endpoints = shard_endpoints_opt.unwrap();
        if shard_endpoints.len() == 0 {
            error_bail!(
                "shard_endpoints.len() is 0, varname: {}",
                embedding_varname.clone()
            );
        }

        let shard_num = shard_endpoints.len();

        // GpuPsFeature64.
        //
        // Signs need to be distributed to different ps shard, based on the value. Use the last bits
        // to determin ps shard index.
        let mut features: Vec<GpuPsFeature64> = Vec::with_capacity(shard_num);
        features.resize(shard_num, GpuPsFeature64::default());

        // Assign each sign and item_index to corresponding ps.
        for i in 0..total_signs {
            let shard_index = get_target_shard_by_sign(signs[i], shard_num);

            features[shard_index].features.push(signs[i]);
            features[shard_index]
                .item_indices
                .push(item_indexes[i] as i32);
        }

        // Serialize GpuPsFeature64 to string.
        for i in 0..shard_num {
            let mut buf = Vec::new();
            match features[i].encode(&mut buf) {
                Ok(_) => {}
                Err(err) => {
                    error_bail!("gpu_ps_feature64 encode to str failed! error: {}", err);
                }
            }

            res.insert(shard_endpoints[i].clone(), buf);
        }

        Ok(res)
    }

    /// Send batch_id, dense features and labels to channel.
    pub async fn send_to_trainer(&mut self, sample_batch: SampleBatch) -> Result<()> {
        match self.trainer_data_sender.send(sample_batch).await {
            Ok(_) => Ok(()),
            Err(err) => {
                error_bail!("send data to channel for trainer failed! error: {}", err);
            }
        }
    }

    /// Process the sample batch, send feature to ps and trainer.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        loop {
            tokio::select! {
                sample_batch_res = self.sample_batch_receiver.recv() => {
                    match sample_batch_res {
                        Ok(sample_batch) => {
                            self.total_batch += 1;

                            info!(
                                "get one batch! total_batch: {}, batch_id: {}",
                                self.total_batch,
                                sample_batch.batch_id.clone(),
                            );

                            match self.send_to_ps(&sample_batch).await {
                                Ok(_) => {},
                                Err(err) => {
                                    error!("send_to_ps failed! err: {}", err);
                                }
                            }

                            match self.send_to_trainer(sample_batch).await {
                                Ok(_) => {},
                                Err(err) => {
                                    error!("send_to_trainer failed! err: {}", err);
                                }
                            }
                        },
                        Err(err) => {
                            error_bail!("get sample batch failed! error: {}", err);
                        }
                    }
                },
                _ = subsys.on_shutdown_requested() => {
                    info!("FeedSample shutdown!");
                    return Ok(());
                }
            }
        }
    }
}
