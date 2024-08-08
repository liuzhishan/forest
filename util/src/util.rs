use env_logger;
use log::{error, info};

use std::thread;

use anyhow::anyhow;
use anyhow::bail;

use tokio::signal::unix::{signal, SignalKind};
use tonic::{transport::Server, Code, Request, Response, Status};
use tonic_types::{ErrorDetails, StatusExt};

use std::hash::{DefaultHasher, Hash, Hasher};

use std::io::Write;

/// Init log. Set log format.
pub fn init_log() {
    env_logger::builder()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] {}:{} - {}",
                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .filter_level(log::LevelFilter::Info)
        .init();
}

async fn wait_for_signal_impl() {
    // Infos here:
    // https://www.gnu.org/software/libc/manual/html_node/Termination-Signals.html
    let mut signal_terminate = signal(SignalKind::terminate()).unwrap();
    let mut signal_interrupt = signal(SignalKind::interrupt()).unwrap();

    tokio::select! {
        _ = signal_terminate.recv() => tracing::debug!("Received SIGTERM."),
        _ = signal_interrupt.recv() => tracing::debug!("Received SIGINT."),
    };
}

pub async fn wait_for_signal() {
    wait_for_signal_impl().await
}

// bail!($fmt, $($arg)*)
#[macro_export]
macro_rules! error_bail {
    ($msg:literal $(,)?) => {
        error!($msg);
        bail!($msg)
    };
    ($err:expr $(,)?) => {
        error!($err);
        bail!(err)
    };
    ($fmt:expr, $($arg:tt)*) => {
        error!($fmt, $($arg)*);
        bail!($fmt, $($arg)*)
    };
}

pub fn send_error_response<T>(err_details: ErrorDetails) -> Result<Response<T>, Status> {
    let status = Status::with_error_details(
        Code::InvalidArgument,
        "request cotains invalid argumetns",
        err_details,
    );

    return Err(status);
}

pub fn compute_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
