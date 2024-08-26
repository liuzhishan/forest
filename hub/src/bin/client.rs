use anyhow::{bail, Result};
use grpc::sniper::EmbeddingTable;
use grpc::sniper::FeatureList;
use grpc::sniper::HdfsSrc;
use grpc::sniper::Role;
use grpc::sniper::SparseFieldInfo;
use grpc::sniper::StartSampleOption;
use grpc::sniper::TensorProto;
use log::info;

use clap::Parser;

use grpc::sniper::sniper_client::SniperClient;
use grpc::sniper::HelloRequest;
use grpc::sniper::TensorMessage;

use prost_types::Any;
use util::Flags;

/// Test say_hello.
async fn test_hello(client: &mut SniperClient<tonic::transport::Channel>) -> Result<()> {
    let request = tonic::Request::new(HelloRequest {
        name: "Tonic".into(),
    });

    info!("Sending request to gRPC Server...");
    let response = client.say_hello(request).await?;

    info!("RESPONSE={:?}", response);

    Ok(())
}

/// Test StartSample.
async fn test_start_sample(
    flags: &Flags,
    client: &mut SniperClient<tonic::transport::Channel>,
) -> Result<()> {
    let mut start_sample_option = StartSampleOption::default();

    start_sample_option.batch_size = 4;
    start_sample_option.parallel = 1;
    start_sample_option.need_batch = true;

    if flags.dirname.is_none() {
        bail!("missing dirname argument!");
    }

    if flags.filename.is_none() {
        bail!("missing filename argument!");
    }

    // ps_endpoints
    // TODO: use real ps.
    start_sample_option
        .ps_eps
        .push(String::from("http://[::1]:50062"));

    // HdfsSrc
    let mut hdfs_src = HdfsSrc::default();
    hdfs_src.dir = flags.dirname.as_ref().unwrap().clone();
    hdfs_src
        .file_list
        .push(flags.filename.as_ref().unwrap().clone());

    info!("read dirname from args: {}", hdfs_src.dir.clone());

    hdfs_src.file_list.iter().for_each(|x| {
        info!("read filename from args: {}", x);
    });

    start_sample_option.hdfs_src = Some(hdfs_src);

    // FeatureList
    // TODO: parse feature file.
    let mut feature_list = FeatureList::default();

    feature_list.sparse_field_count = vec![16];
    feature_list.sparse_hash_size = vec![100001];
    feature_list.sparse_field_index = vec![0];
    feature_list.dense_field_count = vec![2];
    feature_list.sparse_class_names = vec![String::from("ExtractUserViewLikePhotoLabel")];
    feature_list.dense_class_names = vec![String::from("ExtractUserAdLpsNumExtendEcomDense")];

    let mut sparse_field_info = SparseFieldInfo::default();
    sparse_field_info.class_name = String::from("ExtractUserViewLikePhotoLabel");
    sparse_field_info.prefix = 0;
    sparse_field_info.index = 0;
    sparse_field_info.size = 100001;
    sparse_field_info.valid = true;
    sparse_field_info.slot = 0;

    feature_list.field_list.push(sparse_field_info);

    feature_list.sparse_emb_table = vec![String::from("embedding_0")];

    start_sample_option.feature_list = Some(feature_list);

    start_sample_option.dense_total_size = 2;

    let mut embedding_table = EmbeddingTable::default();

    embedding_table.name = String::from("embedding_0");
    embedding_table.dim = 16;
    embedding_table.capacity = 0;
    embedding_table.load = 1.0;
    embedding_table.hash_bucket_size = 100001;
    embedding_table.fields = vec![0];

    start_sample_option.emb_tables.push(embedding_table);

    let options = Any::from_msg(&start_sample_option)?;

    let tensor_message = TensorMessage {
        role: Role::Hub.into(),
        role_id: 1,
        seq_id: 2,
        varname: String::from("a"),
        options: Some(options),
        tensor1: Some(TensorProto::default()),
        tensor2: Some(TensorProto::default()),
    };

    let request = tonic::Request::new(tensor_message);

    info!("client.start_sample before!");

    let response = client.start_sample(request).await?;

    info!("client.start_sample done!");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let flags = Flags::parse();

    util::init_log();

    let mut client = SniperClient::connect("http://[::1]:50052").await?;

    test_start_sample(&flags, &mut client).await?;

    Ok(())
}
