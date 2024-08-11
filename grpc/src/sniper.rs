use anyhow::{anyhow, Result};
use prost::Name;

use tensorflow::{Tensor, TensorType};

tonic::include_proto!("sniper");

impl TensorShapeProto {
    pub fn new(dim: &Vec<i64>) -> Self {
        Self { dim: dim.clone() }
    }
}

/// TensorProto definition:
///
/// message TensorProto {
///   DataType dtype = 1;
///   TensorShapeProto tensor_shape = 2;
///   bytes tensor_content = 3;
///   repeated float float_val = 4;
///   repeated uint64 uint64_val = 5;
/// }
impl TensorProto {
    /// Construct TensorProto from Vec.
    pub fn with_vec<T: TensorType>(dtype: i32, dim: &Vec<i64>, values: &Vec<T>) -> Self {
        // shape.
        let tensor_shape = Some(TensorShapeProto { dim: dim.clone() });

        /// bytes.
        let (head, body, tail) = unsafe { values.align_to::<u8>() };

        assert!(head.is_empty());
        assert!(tail.is_empty());

        let tensor_content = body.to_vec();

        Self {
            dtype: dtype.into(),
            tensor_shape,
            tensor_content,
            float_val: Vec::new(),
            uint64_val: Vec::new(),
        }
    }
}

impl TensorMessage {}

impl VoidMessage {}

impl Name for StartSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "StartSampleOption";
}

impl Name for FeedSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "FeedSampleOption";
}

impl Name for ReadSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "ReadSampleOption";
}
