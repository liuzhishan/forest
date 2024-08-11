use std::time::Duration;

use anyhow::bail;
use log::{error, info};

use tokio_graceful_shutdown::{SubsystemBuilder, Toplevel};
use tonic::{transport::Server, Code, Request, Response, Status};
use tonic_types::{ErrorDetails, StatusExt};

use dashmap::DashMap;

use grpc::sniper::sniper_ps_server::{SniperPs, SniperPsServer};
use grpc::sniper::{
    start_sample_option, DataType, HelloRequest, HelloResponse, StartSampleOption, TensorMessage,
    VoidMessage,
};

use grpc::sniper::{TensorProto, TensorShapeProto};
use util::send_error_response;

/// Ps server.
///
/// Start ps for processing parameters.
#[derive(Debug)]
pub struct Ps {
}

impl Ps {
    pub fn new() -> Self {
        // TODO
        Self {}
    }
}

#[tonic::async_trait]
impl SniperPs for Ps {
    /// Create ps vars embedding table and dense var parameters in ps.
    async fn create(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Freeze graph.
    async fn freeze(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// FeedSample.
    async fn feed_sample(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        let request_inner = request.into_inner();

        info!(
            "[SniperPs.feed_sample] get feed_sample request: {:?}",
            request_inner
        );

        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Push.
    async fn push(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Pull
    async fn pull(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        // TODO
        let mut response = TensorMessage::default();

        Ok(Response::new(response))
    }

    /// EmbeddingLookup
    async fn embedding_lookup(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<TensorMessage>, Status> {
        // TODO
        let mut response = TensorMessage::default();

        Ok(Response::new(response))
    }

    /// PushGrad.
    async fn push_grad(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Save.
    async fn save(&self, request: Request<TensorMessage>) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Restore.
    async fn restore(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Complete.
    async fn complete(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }

    /// Heartbeat.
    async fn heartbeat(
        &self,
        request: Request<TensorMessage>,
    ) -> Result<Response<VoidMessage>, Status> {
        // TODO
        let mut response = VoidMessage::default();

        Ok(Response::new(response))
    }
}
