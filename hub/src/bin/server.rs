use anyhow::Result;
use log::info;

use local_ip_address::local_ip;
use tonic::transport::Server;

use grpc::sniper::sniper_server::SniperServer;
use hub::request_handler::Hub;
use hub::tool::HUB_SERVER_PORT;
use util::wait_for_signal;

async fn serve() {
    let my_local_ip = local_ip().unwrap();

    let addr = format!("{}:{}", my_local_ip, HUB_SERVER_PORT)
        .parse()
        .unwrap();
    let hub = Hub::new();

    let signal = wait_for_signal();

    info!(
        "Starting gRPC Server..., ip: {}, port: {}",
        my_local_ip, HUB_SERVER_PORT
    );

    let limit = 20 * 1024 * 1024;

    Server::builder()
        .add_service(
            SniperServer::new(hub)
                .max_decoding_message_size(limit)
                .max_encoding_message_size(limit),
        )
        .serve_with_shutdown(addr, signal)
        .await
        .unwrap();
}

// Runtime to run our server
fn main() -> Result<()> {
    util::init_log();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(serve());

    Ok(())
}
