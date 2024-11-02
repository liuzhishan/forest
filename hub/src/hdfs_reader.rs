use anyhow::Result;
use log::{error, info};
use util::histogram::record_time;

use std::io::{BufRead, BufReader};

use coarsetime::Instant;
use hdrs::ClientBuilder;
use tokio_graceful_shutdown::SubsystemHandle;

use util::histogram::{Histogram, HistogramType};

/// Read data from hdfs.
pub struct HdfsReader {
    /// Filenames to read.
    filenames: Vec<String>,

    /// Send features to channel.
    line_sender: async_channel::Sender<String>,

    /// Histogram statistics.
    histogram: Histogram,

    /// Id in threads.
    id: i32,
}

impl HdfsReader {
    pub fn new(
        filenames: &Vec<String>,
        line_sender: async_channel::Sender<String>,
        histogram: Histogram,
        id: i32,
    ) -> Self {
        Self {
            filenames: filenames.iter().cloned().collect(),
            line_sender,
            histogram,
            id,
        }
    }

    /// It's slow to check all hdfs path exists. It must be verified in trainer. Hub would
    /// assume that all the paths are exists.
    pub async fn init(&mut self) -> bool {
        true
    }

    /// Read from hdfs paths, send line to channel.
    pub async fn run(mut self, _subsys: SubsystemHandle) -> Result<()> {
        let fs = ClientBuilder::new(&"default").connect()?;

        for filename in self.filenames.iter() {
            info!(
                "open hdfs file, reader id: {}, filename: {}",
                self.id,
                filename.clone()
            );

            let f = fs.open_file().read(true).open(filename)?;
            let reader = BufReader::new(f);

            let mut last = Instant::now();

            for line_result in reader.lines() {
                let line = line_result?;

                record_time(
                    &mut self.histogram,
                    HistogramType::HubReadMessage,
                    &mut last,
                );

                match self.line_sender.send(line).await {
                    Ok(_) => {}
                    Err(err) => {
                        error!("send line error! error: {}", err);
                    }
                }
            }

            info!(
                "read hdfs done, reader id: {}, file: {}",
                self.id,
                filename.clone()
            );
        }

        info!("hdfs reader done successfully! reader id: {}", self.id);

        Ok(())
    }
}
