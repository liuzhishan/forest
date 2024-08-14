use anyhow::Result;
use grpc::sniper::sniper_ps_client::SniperPsClient;
use log::info;

pub fn start_ps() {
    info!("start ps!");
}

/// For test.
pub async fn get_ps_default_client() -> Result<SniperPsClient<tonic::transport::Channel>> {
    match SniperPsClient::connect("http://[::1]:50062").await {
        Ok(client) => Ok(client),
        Err(err) => Err(err.into()),
    }
}

pub async fn get_ps_client(
    ps_endpoint: String,
) -> Result<SniperPsClient<tonic::transport::Channel>> {
    match SniperPsClient::connect(ps_endpoint).await {
        Ok(client) => Ok(client),
        Err(err) => Err(err.into()),
    }
}
