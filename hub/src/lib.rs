//! IO module.
//!
//! This module contains the data processing logic.
//!
//! The main component are:
//! - `task`: Task node for data processing logic.
//! - `sample`: Data sample for representing the input data for training.
//! - `pipeline`: Pipeline of composed task nodes for processing specifical data format.
//! - `hub_server`: Main logic of hub, starting all task nodes.
//! - `grpc handler`: Grpc handler for processing data.

mod tool;

mod sample;

mod batch_assembler;
mod hdfs_reader;
mod local_reader;
mod task;

mod request_handler;
