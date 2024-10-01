//! Parameter server.
//!
//! Store, query and update the embedding parameters and dense parameters.
//! Since embedding parameter and dense parameters has very different interface
//! and use case, We dont use generic data structure. Instend we use concrete
//! data structure and variable_manager for them.

#![feature(portable_simd)]
use core::simd::prelude::*;

pub mod tool;
pub use tool::get_ps_client;

pub mod request_handler;
pub use request_handler::Ps;

mod arc_unsafe_slice;
mod arc_unsafe_vec;
mod dense;
mod embedding;
mod env;
mod scheduler;
mod sign_converter;
mod variable_manager;

mod checkpoint;
