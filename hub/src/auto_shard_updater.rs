use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::{select, sync::broadcast};
use util::FeaturePlacement;

use tokio_graceful_shutdown::{IntoSubsystem, SubsystemBuilder, SubsystemHandle, Toplevel};

use log::{error, info};

/// Auto shard updater.
///
/// Receive new ps shard from channel, and update placement.
pub struct AutoShardUpdater {
    /// Placement.
    placement: Arc<Mutex<FeaturePlacement>>,

    /// Receiver.
    receiver: broadcast::Receiver<Vec<Vec<i32>>>,
}

impl AutoShardUpdater {
    pub fn new(
        placement: Arc<Mutex<FeaturePlacement>>,
        receiver: broadcast::Receiver<Vec<Vec<i32>>>,
    ) -> Self {
        Self {
            placement,
            receiver,
        }
    }

    pub async fn run(mut self, subsystem: SubsystemHandle) -> Result<()> {
        loop {
            select! {
                ps_shard = self.receiver.recv() => {
                    match ps_shard {
                        Ok(shard) => {
                            let mut placement = self.placement.lock().unwrap();
                            placement.update_ps_shard(shard)?;
                        }
                        Err(err) => {
                            error!("auto shard updater recv ps shard failed! err: {}", err);
                        }
                    }
                }
                _ = subsystem.on_shutdown_requested() => {
                    info!("auto shard updater shutdown requested");
                    return Ok(());
                }
            }
        }
    }
}
