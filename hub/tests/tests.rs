use log::{error, info};

use rand::RngCore;
use std::sync::Once;

use anyhow::Result;
use tokio::time::{sleep, Duration};
use tokio_graceful_shutdown::{SubsystemBuilder, SubsystemHandle, Toplevel};

use hub::local_reader::{self, LocalReader};
use util::init_log;

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        init_log();
    });
}

#[test]
fn test_setup() {
    setup();
}

#[test]
fn test_rand() {
    setup();

    let v: u64 = rand::thread_rng().next_u64();
    info!("rand u64: {}", v);
}

/// Test local reader.
///
/// Read data from local file, which contains SimpleFeatures in base64 format.
#[tokio::test]
async fn test_local_reader() -> Result<()> {
    setup();

    let (s, r) = async_channel::bounded(10);

    let filenames = &vec![String::from(
        "/home/liuzhishan/ast/data/dsp_conv_simple_features_head_100.txt",
    )];

    let local_reader = LocalReader::new(filenames, s);

    Toplevel::new(|s| async move {
        s.start(SubsystemBuilder::new("local_reader", |a| {
            local_reader.run(a)
        }));
    })
    .catch_signals()
    .handle_shutdown_requests(Duration::from_millis(1000))
    .await
    .map_err(Into::into)
}
