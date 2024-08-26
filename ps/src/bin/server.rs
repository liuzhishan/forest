use anyhow::Result;
use log::info;
use tonic::transport::Server;
use util::wait_for_signal;

use grpc::sniper::sniper_server::SniperServer;
use ps::request_handler::Ps;

#[tokio::main]
async fn main() -> Result<()> {
    util::init_log();

    let addr = "[::1]:50062".parse()?;
    let ps = Ps::new();

    let signal = wait_for_signal();

    info!("Starting gRPC Server...");
    Server::builder()
        .add_service(SniperServer::new(ps))
        .serve_with_shutdown(addr, signal)
        .await?;

    Ok(())
}
