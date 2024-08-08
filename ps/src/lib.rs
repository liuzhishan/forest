//! Parameter server.

mod tool;
pub use tool::get_ps_client;

pub mod request_handler;
pub use request_handler::Ps;
