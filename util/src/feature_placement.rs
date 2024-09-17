use log::{error, info};

use hashbrown::HashMap;

use grpc::sniper::EmbeddingTable;

use crate::simple_string_to_int_hash;

use super::compute_hash;

/// Strategy to place var to different ps.
///
/// The goal is to minimize the total load of all ps.
///
/// Each var is assigned one or multi ps.
///
/// The first version is simple. Use varname to compute hash, and use hash % ps_count
/// as the result. It's a enough-to-use strategy. Futher optimization can be done conserding
/// unique sign count, real time spend, and other factors.
pub struct FeaturePlacement {
    pub vars: Vec<EmbeddingTable>,
    pub ps_endpoints: Vec<String>,
    max_shard: i32,
    ps_shard: HashMap<String, Vec<String>>,
}

impl FeaturePlacement {
    pub fn new(vars: &Vec<EmbeddingTable>, ps_endpoints: &Vec<String>) -> Self {
        let max_shard = 1;
        let ps_shard = Self::compute_ps_shard(vars, ps_endpoints);

        Self {
            vars: vars.clone(),
            ps_endpoints: ps_endpoints.clone(),
            max_shard,
            ps_shard,
        }
    }

    /// Compute each ps shard based on varname hash.
    fn compute_ps_shard(
        vars: &Vec<EmbeddingTable>,
        ps_endpoints: &Vec<String>,
    ) -> HashMap<String, Vec<String>> {
        let mut res = HashMap::new();

        let total_ps = ps_endpoints.len();

        for var in vars.iter() {
            let h = simple_string_to_int_hash(var.name.as_str());
            let index = (h % total_ps as u64) as usize;

            info!(
                "placement, insert ps shard, varname: {}, index: {}, ps_endpoint: {}",
                var.name.clone(),
                index,
                ps_endpoints[index].clone(),
            );

            res.insert(var.name.clone(), vec![ps_endpoints[index].clone()]);
        }

        res
    }

    /// Get ps_endpoints by varname.
    pub fn get_emb_placement(&self, varname: &String) -> Option<&Vec<String>> {
        self.ps_shard.get(varname)
    }
}
