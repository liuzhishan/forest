//! Dense parameter of network.

use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use grpc::sniper::PullOption;
use log::{error, info};
use std::sync::Arc;
use std::sync::Mutex;
use util::histogram;
use util::histogram::WithHistogram;
use util::{error_bail, histogram::Histogram};

/// Store dense parameters.
///
/// Such as dense feature, network parameters. The name and dims must be provied on construction.
pub struct DenseVariable {
    /// Varname.
    pub varname: String,

    /// Dims
    pub dims: Vec<usize>,

    /// Total size of all dims.
    pub total_size: usize,

    /// Vlues.
    pub values: Vec<f32>,

    /// Histogram statistics.
    histogram: Arc<Mutex<Histogram>>,
}

impl WithHistogram for DenseVariable {
    fn with_histogram(histogram: Histogram) -> Self {
        Self {
            varname: String::from(""),
            dims: Vec::new(),
            total_size: 0,
            values: Vec::new(),
            histogram: Arc::new(Mutex::new(histogram)),
        }
    }
}

impl DenseVariable {
    /// Intialize all values to 0.0 in new.
    pub fn new(varname: &String, dims: &Vec<usize>, histogram: Arc<Mutex<Histogram>>) -> Self {
        let total_size = dims.iter().fold(1, |acc, x| acc * x);
        let values: Vec<f32> = Vec::with_capacity(total_size);

        Self {
            varname: varname.clone(),
            dims: dims.clone(),
            total_size,
            values,
            histogram,
        }
    }

    /// Save the parameters to self.values.
    pub fn push(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.total_size {
            error_bail!(
                "values.len() != self.total_size, values.len(): {}, self.total_size: {}",
                values.len(),
                self.total_size,
            );
        }

        self.values = values.to_vec();

        Ok(())
    }

    /// Save values slice to position if start.
    ///
    /// Need to check overflow.
    pub fn push_from_slice(&mut self, values: &[f32], start: usize) -> Result<()> {
        // Index should not be bigger than total_size.
        if start + values.len() > self.total_size {
            return Err(anyhow!(
                "out of range, start + values.len() > self.total_size, start: {}, values.len(): {}, total_size: {}",
                start,
                values.len(),
                self.total_size,
            ));
        }

        // This only happens the first time some values are pushed.
        if start + values.len() >= self.values.len() {
            self.values.resize(self.total_size, 0.0);
        }

        for i in 0..values.len() {
            self.values[i + start] = values[i];
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
