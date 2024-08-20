use hashbrown::HashMap;

#[derive(Default)]
pub struct Scheduler {}

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
        true
    }

    pub fn start(&mut self) {}

    pub fn stop(&mut self) {}
}
