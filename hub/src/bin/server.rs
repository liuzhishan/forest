use log::info;
use tonic::{transport::Server, Request, Response, Status};

use env_logger;

use grpc::sniper::sniper_server::{Sniper, SniperServer};
use grpc::sniper::{DataType, HelloRequest, HelloResponse, TensorMessage, VoidMessage};
use grpc::sniper::{TensorProto, TensorShapeProto};

/// rpc server.
#[derive(Debug, Default)]
pub struct MySniper {}

#[tonic::async_trait]
impl Sniper for MySniper {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloResponse>, Status> {
        info!("Received request from: {:?}", request);

        let response = HelloResponse {
            message: format!("Hello {}!", request.into_inner().name).into(),
        };

        Ok(Response::new(response))
    }

    async fn create(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        info!("Received create request: {:?}", request);

        let response = VoidMessage {};

        Ok(Response::new(response))
    }

    async fn pull(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        info!("Received create request: {:?}", request);

        let mut response = TensorMessage::new();

        response.varname = String::from("pull");

        Ok(Response::new(response))
    }
}

// Runtime to run our server
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    util::init_log();

    let addr = "[::1]:50052".parse()?;
    let sniper = MySniper::default();

    info!("Starting gRPC Server...");
    Server::builder()
        .add_service(SniperServer::new(sniper))
        .serve(addr)
        .await?;

    Ok(())
}
