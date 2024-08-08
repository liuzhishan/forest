use log::info;

use tonic::transport::Server;

use grpc::sniper::sniper_hub_server::SniperHubServer;
use hub::request_handler::Hub;
use util::wait_for_signal;

// Runtime to run our server
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    util::init_log();

    let addr = "[::1]:50052".parse()?;
    let hub = Hub::new();

    let signal = wait_for_signal();

    info!("Starting gRPC Server...");
    Server::builder()
        .add_service(SniperHubServer::new(hub))
        .serve_with_shutdown(addr, signal)
        .await?;

    Ok(())
}
