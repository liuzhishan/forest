use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// All command line flags.
#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Flags {
    /// Dirname.
    #[arg(long)]
    pub dirname: Option<String>,

    /// Filename.
    #[arg(long)]
    pub filename: Option<String>,
}
