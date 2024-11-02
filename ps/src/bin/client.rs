use anyhow::Result;
use grpc::sniper::{sniper_client::SniperClient, TensorMessage};
use log::info;
use ps::tool::get_ps_default_client;

/// Test pull.
async fn test_pull(_client: &mut SniperClient<tonic::transport::Channel>) -> Result<()> {
    let _request = tonic::Request::new(TensorMessage::default());

    info!("Sending request to gRPC Server...");
    // let response = client.pull(request).await?;

    // info!("RESPONSE={:?}", response);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    util::init_log();

    let mut client = get_ps_default_client().await?;

    test_pull(&mut client).await?;

    Ok(())
}
