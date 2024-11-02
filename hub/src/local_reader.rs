use anyhow::Result;
use log::{error, info};

use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::path::Path;

use tokio_graceful_shutdown::SubsystemHandle;

use util::histogram::Histogram;

/// Read data from local.
pub struct LocalReader {
    /// Filenames to read.
    filenames: Vec<String>,

    /// Send features to channel.
    line_sender: async_channel::Sender<String>,

    /// Histogram statistics.
    histogram: Histogram,
}

impl LocalReader {
    pub fn new(
        filenames: &Vec<String>,
        line_sender: async_channel::Sender<String>,
        histogram: Histogram,
    ) -> Self {
        Self {
            filenames: filenames.iter().cloned().collect(),
            line_sender,
            histogram,
        }
    }

    /// Check all filenames exists.
    pub async fn init(&mut self) -> bool {
        if self.filenames.len() > 0 {
            if self.filenames.iter().all(|x| Path::new(x).exists()) {
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Read from filenames, and send line to the channel.
    pub async fn run(self, _subsys: SubsystemHandle) -> Result<()> {
        info!("LocalReader run start");

        for (_, filename) in self.filenames.iter().enumerate() {
            let file = File::open(filename)?;
            let reader = BufReader::new(file);

            for line_result in reader.lines() {
                let line = line_result?;

                match self.line_sender.send(line).await {
                    Ok(_) => {}
                    Err(err) => {
                        error!("send line error! error: {}", err);
                    }
                }
            }
        }

        info!("LocalReader run done");

        Ok(())
    }
}
