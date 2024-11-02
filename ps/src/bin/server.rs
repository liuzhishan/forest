use anyhow::Result;
use local_ip_address::local_ip;
use log::info;
use util::wait_for_signal;

use tonic::transport::Server;

use grpc::sniper::sniper_server::SniperServer;
use ps::request_handler::Ps;
use ps::tool::PS_SERVER_PORT;

async fn serve() {
    let ps = Ps::new();

    let my_local_ip = local_ip().unwrap();
    let addr = format!("{}:{}", my_local_ip.clone(), PS_SERVER_PORT)
        .parse()
        .unwrap();

    let signal = wait_for_signal();

    let limit = 20 * 1024 * 1024;

    info!(
        "Starting gRPC Server..., ip: {}, port: {}",
        my_local_ip, PS_SERVER_PORT
    );

    Server::builder()
        .add_service(
            SniperServer::new(ps)
                .max_decoding_message_size(limit)
                .max_encoding_message_size(limit),
        )
        .serve_with_shutdown(addr, signal)
        .await
        .unwrap();
}

/// This approach has some other problems.
///
/// 1. Every instance of `Ps` would have all the `parameter` state.
/// 2. `hub` send `batch_id` to `ps` and `trainer`, then `trainer` will send the `batch_id`
///     to `ps` to get `Embedding` parameters. But the request from `hub` and `trainer` may
///     not be processed by the same thread, so the `batch_id` would be not found in `Embedding`
///     variables.
#[allow(dead_code)]
fn multi_thread_serve() {
    let mut handlers = Vec::new();
    for _i in 0..num_cpus::get() {
        let h = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(serve());
        });

        handlers.push(h);
    }

    for h in handlers {
        h.join().unwrap();
    }
}

fn main() -> Result<()> {
    util::init_log();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(serve());

    Ok(())
}
