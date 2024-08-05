use std::sync::Mutex;

use log::info;

/// GRPC client.
pub struct GrpcClient {
    /// Role id.
    pub role_id: i32,
}
