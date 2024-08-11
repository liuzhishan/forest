use anyhow::Result;
use grpc::sniper::{sniper_ps_client::SniperPsClient, TensorMessage};
use log::{error, info};

/// Test pull.
async fn test_pull(client: &mut SniperPsClient<tonic::transport::Channel>) -> Result<()> {
    let request = tonic::Request::new(TensorMessage::default());

    info!("Sending request to gRPC Server...");
    // let response = client.pull(request).await?;

    // info!("RESPONSE={:?}", response);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    util::init_log();

    let mut client = SniperPsClient::connect("http://[::1]:50062").await?;

    test_pull(&mut client).await?;

    Ok(())
}
