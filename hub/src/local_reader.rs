use anyhow::anyhow;
use anyhow::bail;
use anyhow::Error;
use anyhow::Result;
use log::{error, info};
use prost::Message;

use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::path::Path;

use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};

use grpc::sniper::SimpleFeatures;

/// Read data from local.
pub struct LocalReader {
    /// Filenames to read.
    filenames: Vec<String>,

    /// Send features to channel.
    line_sender: async_channel::Sender<String>,
}

impl LocalReader {
    pub fn new(filenames: &Vec<String>, line_sender: async_channel::Sender<String>) -> Self {
        Self {
            filenames: filenames.iter().cloned().collect(),
            line_sender,
        }
    }

    /// Check all filenames exists.
    pub fn init(&mut self) -> bool {
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

    /// Read from filenames, parse each line to SimpleFeatures, and send to the feature_channel.
    pub async fn run(mut self, subsys: SubsystemHandle) -> Result<()> {
        info!("LocalReader run start");

        for (_, filename) in self.filenames.iter().enumerate() {
            let file = File::open(filename)?;
            let reader = BufReader::new(file);

            for line_result in reader.lines() {
                let line = line_result?;

                info!("LocalReader send line, line.size(): {}", line.len());

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