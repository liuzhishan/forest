use anyhow::{anyhow, Result};
use log::{error, info};
use prost::{Message, Name};

use tensorflow::{Tensor, TensorType};

tonic::include_proto!("sniper");

impl StartSampleOption {
    pub fn get_embedding_varnames(&self, sparse_feature_count: usize) -> Vec<String> {
        match self.feature_list.as_ref() {
            Some(x) => x.sparse_emb_table.clone(),
            None => {
                error!("feature_list is None!");
                Vec::new()
            }
        }
    }
}

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

    pub fn as_slice<T: TensorType>(&self) -> &[T] {
        let (head, body, tail) = unsafe { self.tensor_content.align_to::<T>() };

        assert!(head.is_empty());
        assert!(tail.is_empty());

        body
    }
}

impl TensorMessage {}

impl VoidMessage {}

impl Name for StartSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "StartSampleOption";
}

impl Name for CreateOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "CreateOption";
}

impl Name for FeedSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "FeedSampleOption";
}

impl Name for ReadSampleOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "ReadSampleOption";
}

impl Name for FreezeOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "FreezeOption";
}

impl Name for PushOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "PushOption";
}

impl Name for PullOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "PullOption";
}

impl Name for EmbeddingLookupOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "EmbeddingLookupOption";
}

impl Name for PushGradOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "PushGradOption";
}

impl Name for SaveOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "SaveOption";
}

impl Name for RestoreOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "RestoreOption";
}

impl Name for HeartbeatOption {
    const PACKAGE: &'static str = "sniper";
    const NAME: &'static str = "HeartbeatOption";
}
