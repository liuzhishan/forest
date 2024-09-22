use anyhow::Result;
use grpc::sniper::sniper_client::SniperClient;
use log::info;

use local_ip_address::local_ip;
use util::MESSAGE_LIMIT;

pub const PS_SERVER_PORT: i32 = 34000;

pub fn start_ps() {
    info!("start ps!");
}

pub async fn get_ps_default_client() -> Result<SniperClient<tonic::transport::Channel>> {
    let my_local_ip = local_ip()?;

    match SniperClient::connect(format!("http://{}:{}", my_local_ip, PS_SERVER_PORT)).await {
        Ok(client) => Ok(client
            .max_decoding_message_size(MESSAGE_LIMIT)
            .max_encoding_message_size(MESSAGE_LIMIT)),
        Err(err) => Err(err.into()),
    }
}

pub async fn get_ps_client(
    ps_endpoint: &String,
) -> Result<SniperClient<tonic::transport::Channel>> {
    match SniperClient::connect(format!("http://{}", ps_endpoint.clone())).await {
        Ok(client) => Ok(client
            .max_decoding_message_size(MESSAGE_LIMIT)
            .max_encoding_message_size(MESSAGE_LIMIT)),
        Err(err) => Err(err.into()),
    }
}
