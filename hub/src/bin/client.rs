use log::info;
use anyhow::Result;

use api::sniper::HelloRequest;
use api::sniper::sniper_client::SniperClient;
use api::sniper::TensorMessage;

/// Test say_hello.
async fn test_hello(
    client: &mut SniperClient<tonic::transport::Channel>,
) -> Result<()> {
    let request = tonic::Request::new(HelloRequest {
        name: "Tonic".into(),
    });

    info!("Sending request to gRPC Server...");
    let response = client.say_hello(request).await?;

    info!("RESPONSE={:?}", response);

    Ok(())
}

/// Test pull.
async fn test_pull(
    client: &mut SniperClient<tonic::transport::Channel>,
) -> Result<()> {
    let request = tonic::Request::new(TensorMessage::new());

    info!("Sending request to gRPC Server...");
    let response = client.pull(request).await?;

    info!("RESPONSE={:?}", response);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    util::init_log();

    let mut client = SniperClient::connect("http://[::1]:50052").await?;

    test_hello(&mut client).await?;

    test_pull(&mut client).await?;

    Ok(())
}
