//! IO module.
//!
//! This module contains the data processing logic.
//!
//! The main component are:
//! - `sample`: Data sample for representing the input data for training.
//! - `pipeline`: Pipeline of composed task nodes for processing specifical data format.
//! - `hub_server`: Main logic of hub, starting all task nodes.
//! - `grpc handler`: Grpc handler for processing data.

#![allow(dead_code)]

pub mod tool;

mod sample;

mod batch_assembler;
mod feed_sample;
mod hdfs_reader;
mod start_sample;

mod pipeline;

pub mod auto_shard_updater;
pub mod local_reader;
pub mod request_handler;
