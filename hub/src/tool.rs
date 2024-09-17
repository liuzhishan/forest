use anyhow::Result;

use local_ip_address::local_ip;

use grpc::sniper::sniper_client::SniperClient;
use log::info;
use util::MESSAGE_LIMIT;

pub const HUB_SERVER_PORT: i32 = 35000;

/// For test.
pub async fn get_hub_default_client() -> Result<SniperClient<tonic::transport::Channel>> {
    let my_local_ip = local_ip()?;

    match SniperClient::connect(format!("http://{}:{}", my_local_ip, HUB_SERVER_PORT)).await {
        Ok(client) => Ok(client
            .max_decoding_message_size(MESSAGE_LIMIT)
            .max_encoding_message_size(MESSAGE_LIMIT)),
        Err(err) => Err(err.into()),
    }
}

pub async fn get_hub_client(
    hub_endpoint: String,
) -> Result<SniperClient<tonic::transport::Channel>> {
    match SniperClient::connect(format!("http://{}", hub_endpoint)).await {
        Ok(client) => Ok(client
            .max_decoding_message_size(MESSAGE_LIMIT)
            .max_encoding_message_size(MESSAGE_LIMIT)),
        Err(err) => Err(err.into()),
    }
}
