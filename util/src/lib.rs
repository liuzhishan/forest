#![feature(portable_simd)]
use core::simd::prelude::*;

mod util;
pub use util::*;

mod feature_placement;
pub use feature_placement::FeaturePlacement;

mod flags;
pub use flags::Flags;

mod status;
pub use status::Status;
pub use status::StatusCode;

pub mod histogram;

pub mod simd;
