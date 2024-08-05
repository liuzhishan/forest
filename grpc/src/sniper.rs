tonic::include_proto!("sniper");

impl TensorShapeProto {
    pub fn new() -> Self {
        Self { dim: Vec::new() }
    }
}

impl TensorProto {
    pub fn new() -> Self {
        Self {
            dtype: DataType::DtInvalid.into(),
            tensor_shape: Some(TensorShapeProto::new()),
            tensor_content: Vec::new(),
            float_val: Vec::new(),
            uint64_val: Vec::new(),
        }
    }
}

impl TensorMessage {
    pub fn new() -> Self {
        Self {
            role: Role::Hub.into(),
            role_id: 0,
            seq_id: 1,
            varname: "a".into(),
            options: None,
            tensor1: Some(TensorProto::new()),
            tensor2: Some(TensorProto::new()),
        }
    }
}
