use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::default;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use dashmap::mapref::one::{Ref, RefMut};
use log::{error, info};

use dashmap::DashMap;
use sync_unsafe_cell::SyncUnsafeCell;
use util::error_bail;
use util::histogram;
use util::histogram::WithHistogram;

use crate::arc_unsafe_vec::ArcUnsafeVec;
use crate::dense::DenseVariable;

use util::histogram::Histogram;

use crate::embedding::Embedding;

/// Perfect string hash for a closed set of varnames.
///
/// Each time a varname is added, the index of the varname in `varnames` is stored in mapping.
pub struct VarnameHash {
    /// Varnames.
    varnames: Vec<String>,

    /// Varname index mapping.
    mapping: DashMap<String, usize>,

    /// Where the variable deleted.
    is_deleted: Vec<bool>,
}

impl Default for VarnameHash {
    fn default() -> Self {
        Self {
            varnames: Vec::with_capacity(1000),
            mapping: DashMap::new(),
            is_deleted: Vec::new(),
        }
    }
}

impl VarnameHash {
    /// Construct varnames with `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            varnames: Vec::with_capacity(capacity),
            mapping: DashMap::new(),
            is_deleted: Vec::with_capacity(capacity),
        }
    }

    /// Add a varname, and store the index in `mapping`, then return the index as hash.
    pub fn add_varname(&mut self, varname: &String) -> usize {
        match self.mapping.get(varname) {
            Some(v) => v.value().clone(),
            None => {
                self.varnames.push(varname.clone());
                self.is_deleted.push(false);

                self.mapping
                    .insert(varname.clone(), self.varnames.len() - 1);

                self.varnames.len() - 1
            }
        }
    }

    /// Get hash of a varname, return None if not found.
    pub fn get_hash(&self, varname: &String) -> Option<usize> {
        match self.mapping.get(varname) {
            Some(v) => Some(v.value().clone()),
            None => None,
        }
    }

    /// Just remove the varname from the `mapping` if exists.
    pub fn remove(&mut self, varname: &String) {
        match self.get_hash(varname) {
            Some(v) => {
                if v < self.is_deleted.len() {
                    self.is_deleted[v] = true;
                }
            }
            None => {}
        }
    }
}

/// Variable manager template.
///
/// use the varname as the key, T as the value.
///
/// For example: sparse variable embedding_0 is the sparse embedding of field 0.
///
/// The variable would be access concurrently, we need to support concurrent reading and writing.
/// We use perfect hash and `ArcUnsafeVec<T>` to tackle the performance challenge.
///
/// Why not use `Mutex` for each variable but `SyncUnsafeCell` instead ?
///
/// Add locking to variable would cause much performance issues, since each thread would
/// lock the variable when accessing. To speedup accessing sparse parameters, we use `SyncUnsafeCell`
/// instead, having no lock for each variable. Since signs from different batch are typically
/// not same, so it has little affect on effect of model, but has big performance boosting.
///
/// Why use perfect hash instead of `DashMap` ?
///
/// The reason is that if we use `DashMap`, we need to add lock to the entire map to access
/// in different `spawned` tasks. We cannot add lock to a shard or value, since the value of `DashMap`
/// if a `Ref` which depend on reference of `DashMap`, it cannot be transfer between threads.
///
/// Add lock to entire map would be slow, because there would be only one thread running for all
/// the variables.
///
/// Why use `Arc<Vec>` ?
///
/// We can make perfect hash to mapping to exactly the index of a `Vec`. And for each variable, the
/// index is fixed, then we can add lock to only one variable, and access the variable concurrently
/// in different threads.
///
/// Why not use `Arc<Mutex<Vec>>` ?
///
/// If we use `Arc<Mutex<Vec>>`, we add lock to the entire `Vec`, and it has the same problem as
/// `DashMap` between thredas, Even though `DashMap` has multiple shards in it.
///
/// If `Vec` has no lock, how to we add variable concurrently ?
///
/// The create variable request is sent concurrently. How can we add variable to `Vec` ? We can use
/// another field as the lock, for example, `vars_lock: Mutex<bool>`. When adding variable, we first
/// get lock of `vars_lock`, so we can ensure that there is only one thread pushing to the `Vec`.
/// Once all the variable is constructed at initializing step, we can use `Arc<Vec>` to access them
/// concurrently.
pub struct VariableManager<T: WithHistogram> {
    /// Use `Arc<Vec>` to provide concurrent access to variable between threads.
    ///
    /// Since member of tonic rpc server must be immutable, vars need to be immutable too. We use
    /// `SyncUnsafeCell` to get mutablity.
    vars: ArcUnsafeVec<T>,

    /// Lock for creating variable.
    vars_lock: Mutex<bool>,

    /// Perfect hash for varname.
    varname_hash: Mutex<VarnameHash>,

    /// Max size of vars.
    max_size: usize,

    /// Histogram statistics.
    histogram: Arc<Mutex<Histogram>>,
}

impl<T: WithHistogram> VariableManager<T> {
    pub fn new(histogram: Histogram) -> Self {
        Self {
            vars: ArcUnsafeVec::with_capacity(2000),
            vars_lock: Mutex::new(true),
            varname_hash: Mutex::new(VarnameHash::default()),
            max_size: 2000,
            histogram: Arc::new(Mutex::new(histogram)),
        }
    }

    /// Add a new var.
    pub fn add_new_var(&self, varname: &String, v: T) -> Result<()> {
        let mut hasher = self.varname_hash.lock().unwrap();
        let h = hasher.add_varname(varname);

        let _lock = self.vars_lock.lock().unwrap();

        let size = self.vars.len();

        if h < size {
            let mut var = self.vars.get_element_mut_unchecked(h);
            *var = v;

            Ok(())
        } else if h == size {
            // Must be exactly vars.len().
            self.vars.push(v);

            Ok(())
        } else {
            error_bail!("out of range, h: {}, vars.len(): {}", h, size);
        }
    }

    /// Usr pair (vars, index) to represent a variable.
    pub fn get_var(&self, varname: &String) -> Option<(ArcUnsafeVec<T>, usize)> {
        match self.get_index(varname) {
            Some(index) => Some((self.vars.clone(), index)),
            None => None,
        }
    }

    pub fn get_index(&self, varname: &String) -> Option<usize> {
        let hasher = self.varname_hash.lock().unwrap();
        hasher.get_hash(varname)
    }

    /// Cannot remove from `vars`, because the elements would be shifted to the left
    /// if one element removed. So we just remove the hash.
    pub fn remove(&self, varname: &String) {
        let mut hasher = self.varname_hash.lock().unwrap();
        hasher.remove(varname)
    }

    pub fn vars_arc(&self) -> ArcUnsafeVec<T> {
        self.vars.clone()
    }
}

pub type EmbeddingManager = VariableManager<Embedding>;
pub type DenseManager = VariableManager<DenseVariable>;
