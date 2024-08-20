//! Dense parameter of network.

use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use grpc::sniper::PullOption;
use log::{error, info};
use util::error_bail;

/// Store dense parameters.
///
/// Such as dense feature, network parameters. The name and dims must be provied on construction.
pub struct DenseVariable {
    /// varname
    varname: String,

    /// dims
    dims: Vec<usize>,

    /// vlues.
    values: Vec<f32>,
}

impl DenseVariable {
    /// Intialize all values to 0.0 in new.
    pub fn new(varname: &String, dims: &Vec<usize>) -> Self {
        let total_size = dims.iter().fold(1, |acc, x| acc * x);
        let values: Vec<f32> = Vec::with_capacity(total_size);

        Self {
            varname: varname.clone(),
            dims: dims.clone(),
            values,
        }
    }

    /// Save the parameters to self.values.
    pub fn push(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.values.len() {
            error_bail!(
                "values.len() != self.values.len(), values.len(): {}, self.values.len(): {}",
                values.len(),
                self.values.len(),
            );
        }

        for i in 0..values.len() {
            self.values[i] = values[i];
        }

        Ok(())
    }

    /// Get parameters from self.values.
    pub fn pull(&self, res: &mut Vec<f32>) {
        res.resize(self.values.len(), 0.0);

        for i in 0..self.values.len() {
            res[i] = self.values[i];
        }
    }
}
