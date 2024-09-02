use anyhow::Result;
use grpc::sniper::sniper_client::SniperClient;
use log::info;

pub const PS_SERVER_PORT: i32 = 5800;

pub fn start_ps() {
    info!("start ps!");
}

/// For test.
pub async fn get_ps_default_client() -> Result<SniperClient<tonic::transport::Channel>> {
    match SniperClient::connect(format!("http://[::1]:{}", PS_SERVER_PORT)).await {
        Ok(client) => Ok(client),
        Err(err) => Err(err.into()),
    }
}

pub async fn get_ps_client(ps_endpoint: String) -> Result<SniperClient<tonic::transport::Channel>> {
    match SniperClient::connect(ps_endpoint).await {
        Ok(client) => Ok(client),
        Err(err) => Err(err.into()),
    }
}
