use env_logger;
use log::{error, info};

use std::hash::SipHasher;
use std::thread;

use anyhow::anyhow;
use anyhow::bail;

use std::hash::{DefaultHasher, Hash, Hasher};
use tokio::signal::unix::{signal, SignalKind};

use std::io::Write;

pub const MESSAGE_LIMIT: usize = 20 * 1024 * 1024;

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
        _ = signal_terminate.recv() => {
            info!("Received SIGTERM.");
            tracing::info!("Received SIGTERM.")
        }
        _ = signal_interrupt.recv() => {
            info!("Received SIGINT.");
            tracing::info!("Received SIGINT.")
        }
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

#[inline]
pub fn compute_hash<T: Hash>(t: &T) -> u64 {
    let mut s = SipHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub fn simple_string_to_int_hash(s: &str) -> u64 {
    let mut hash = 0u64;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }

    hash
}

/// Get shard index by sign and shard_num, using bit operation to determin the target shard index.
///
/// The parameter shard_num must be bigger than 0.
///
/// The result shard index is guaranteed to be in range 0..(shard_num - 1).
#[inline]
pub fn get_target_shard_by_sign(sign: u64, shard_num: usize) -> usize {
    (sign & (shard_num - 1) as u64) as usize
}
