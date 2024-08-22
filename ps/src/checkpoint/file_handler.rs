//! File releated handler.

use anyhow::{anyhow, bail, Result};
use std::{fs::File, io::Write};

use log::{error, info};
use std::io::BufReader;

use util::error_bail;

/// Trait for read file.
///
/// Maybe local file or hdfs file.
pub trait FileReader: Sized {
    /// Open a file by filename.
    fn new(filename: &String) -> Result<Self>;

    /// Write content in buf, return the total bytes.
    fn read_line(&mut self) -> Result<String>;
}

pub struct LocalFileReader {
    /// Filename to read.
    filename: String,

    /// Reader.
    reader: BufReader<File>,
}

impl FileReader for LocalFileReader {
    fn new(filename: &String) -> Result<Self> {
        let file = File::open(&filename)?;
        let reader = BufReader::new(file);

        Ok(Self {
            filename: filename.clone(),
            reader,
        })
    }

    fn read_line(&mut self) -> Result<String> {
        Ok("".to_string())
    }
}

/// Trait for write to file.
///
/// Maybe local file or hdfs file.
pub trait FileWriter: Sized {
    /// Open a file by filename.
    fn new(filename: &String) -> Result<Self>;

    /// Write content in buf, return the total bytes.
    fn write(&mut self, buf: &[u8]) -> Result<usize>;

    /// Flush the content in buffer to target.
    fn flush(&mut self) -> Result<()>;
}

/// Write to local file.
pub struct LocalFileWriter {
    /// filename to write.
    filename: String,

    /// Writer.
    writer: File,
}

impl FileWriter for LocalFileWriter {
    fn new(filename: &String) -> Result<Self> {
        match File::create(filename) {
            Ok(writer) => Ok(Self {
                filename: filename.clone(),
                writer,
            }),
            Err(err) => Err(err.into()),
        }
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        match self.writer.write(buf) {
            Ok(x) => Ok(x),
            Err(err) => Err(err.into()),
        }
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        match self.writer.flush() {
            Ok(_) => Ok(()),
            Err(err) => Err(err.into()),
        }
    }
}
