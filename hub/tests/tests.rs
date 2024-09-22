use clap::Parser;
use hashbrown::HashMap;
use log::{error, info};

use rand::RngCore;
use std::borrow::BorrowMut;
use std::sync::Once;
use std::sync::{Arc, Mutex};
use util::histogram::{Histogram, HistogramAggregator, HistogramDetail};

use anyhow::bail;
use anyhow::Result;

use sync_unsafe_cell::SyncUnsafeCell;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tokio_graceful_shutdown::{SubsystemBuilder, SubsystemHandle, Toplevel};

use base64::Engine;
use base64::{engine::general_purpose::STANDARD, read::DecoderReader};

use hdrs::Client;
use hdrs::ClientBuilder;
use std::io::{BufRead, BufReader, Read, Write};

use hub::local_reader::{self, LocalReader};
use util::{compute_hash, error_bail, init_log, simple_string_to_int_hash, Flags};

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

    let flags = Flags::parse();

    if flags.filename.is_none() {
        error_bail!("missing filename in args!");
    }

    let (s, r) = async_channel::bounded(10);

    let filenames = &vec![flags.filename.unwrap()];

    let (histogram_sender, histogram_receiver) = mpsc::channel::<HistogramDetail>(100);
    let histogram = Histogram::new(histogram_sender);

    let local_reader = LocalReader::new(filenames, s, histogram);

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

/// Test hdfs client.
#[test]
fn test_hdfs() -> Result<()> {
    setup();

    let name_node = "default";

    let fs = ClientBuilder::new(&name_node).connect()?;

    let path = format!("/home/ad/liuzhishan/rs/test/test_hdfs.txt");

    let content = "test";

    {
        let meta = fs.metadata(path.as_str());

        info!("path: {}, is_exists: {}", path.clone(), !meta.is_err(),);
    }

    {
        // Write file
        info!("test file write");
        let mut f = fs.open_file().create(true).write(true).open(&path)?;

        for i in 0..5 {
            let s = format!("{} {}\n", content.clone(), i);
            f.write_all(s.as_bytes())?;
        }

        // Flush file
        info!("test file flush");
        f.flush()?;
    }

    {
        // Read file
        info!("test file read");
        let mut f = fs.open_file().read(true).open(&path)?;

        let mut reader = BufReader::new(f);

        let mut s = String::new();

        for line in reader.lines() {
            match line {
                Ok(x) => {
                    info!("read line: {}", x);
                }
                Err(err) => {
                    info!("read line error, err: {}", err);
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Test hdfs base64 reader..
#[test]
fn test_hdfs_base64() -> Result<()> {
    setup();

    let fs = ClientBuilder::new("default").connect()?;

    let path = format!(
        "/home/ad/model_offline/dsp_ctr_lzs_test_v5/model_tf.202409211422/embedding_45.0_0.weight"
    );

    {
        // read base64.
        let f = fs.open_file().read(true).open(&path)?;
        let mut reader = BufReader::new(f);

        for x in reader.lines() {
            match x {
                Ok(line) => {
                    let n = line.len();

                    info!("read line, len: {}", n);

                    let s = match STANDARD.decode(line) {
                        Ok(x) => {
                            info!("decode line success");
                        }
                        Err(err) => {
                            error!("decode line base64 failed! error: {}", err);
                        }
                    };
                }
                Err(err) => {
                    error!("read line error: {}", err);
                }
            }
        }
    }

    Ok(())
}

type ShardedDb = Arc<Vec<Mutex<HashMap<String, String>>>>;

pub fn new_sharded_db(num_shards: usize) -> ShardedDb {
    let mut db = Vec::with_capacity(num_shards);

    for _ in 0..num_shards {
        db.push(Mutex::new(HashMap::new()));
    }

    Arc::new(db)
}

/// Test arc.
#[tokio::test]
async fn test_arc() -> Result<()> {
    setup();

    let n = 10;
    let db = new_sharded_db(n);

    for i in 0..n {
        info!("test_arc, i: {}", i);

        let x = db.clone();

        tokio::spawn(async move {
            let mut shard = x[i].lock().unwrap();
            shard.insert("a".to_string(), "a".to_string());
        });
    }

    Ok(())
}

/// Test UnsafeCell.
#[tokio::test]
async fn test_unsafe_cell() -> Result<()> {
    setup();

    let vec: Arc<SyncUnsafeCell<Vec<i32>>> = Arc::new(SyncUnsafeCell::new(Vec::new()));

    let mut tasks = Vec::new();

    for i in 0..5 {
        let vec_clone = vec.clone();

        tasks.push(tokio::spawn(async move {
            unsafe {
                let v = &mut *vec_clone.as_ref().get();

                v.push(i);
                v.push(i + 10);

                info!("i: {}, size: {}, values: {:?}", i, v.len(), v.clone());
            }
        }));
    }

    for task in tasks {
        task.await;
    }

    Ok(())
}

#[test]
fn test_hash() {
    setup();

    let s = String::from("embedding_0");

    info!("s: {}, hash: {}", s, simple_string_to_int_hash(&s));
}
