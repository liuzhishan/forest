//! File releated handler.

use anyhow::Result;
use std::io::BufRead;
use std::{fs::File, io::Write};

use std::io::BufReader;

#[cfg(feature = "hdfs")]
use hdrs::ClientBuilder;

/// Trait for read file.
///
/// Maybe local file or hdfs file.
pub trait FileReader {
    type Reader: BufRead;

    /// Open a file by filename.
    fn get_reader(filename: &String) -> Result<Self::Reader>;
}

/// Local file reader.
pub struct LocalFileReader {}

impl FileReader for LocalFileReader {
    type Reader = BufReader<File>;

    fn get_reader(filename: &String) -> Result<BufReader<File>> {
        let file = File::open(&filename)?;
        Ok(BufReader::new(file))
    }
}

/// Hdfs file reader.
#[cfg(feature = "hdfs")]
pub struct HdfsFileReader {}

#[cfg(feature = "hdfs")]
impl FileReader for HdfsFileReader {
    type Reader = BufReader<hdrs::File>;

    fn get_reader(filename: &String) -> Result<BufReader<hdrs::File>> {
        let fs = ClientBuilder::new(&"default").connect()?;
        let file = fs.open_file().read(true).open(filename)?;

        Ok(BufReader::new(file))
    }
}

/// Trait for write to file.
///
/// Maybe local file or hdfs file.
pub trait FileWriter {
    type File: Write;

    /// Open a file by filename.
    fn get_writer(filename: &String) -> Result<Self::File>;
}

/// Write to local file.
pub struct LocalFileWriter {}

impl FileWriter for LocalFileWriter {
    type File = File;

    fn get_writer(filename: &String) -> Result<Self::File> {
        match File::create(filename) {
            Ok(writer) => Ok(writer),
            Err(err) => Err(err.into()),
        }
    }
}

/// Write to hdfs file.
#[cfg(feature = "hdfs")]
pub struct HdfsFileWriter {}

#[cfg(feature = "hdfs")]
impl FileWriter for HdfsFileWriter {
    type File = hdrs::File;

    fn get_writer(filename: &String) -> Result<Self::File> {
        let fs = ClientBuilder::new(&"default").connect()?;
        let writer = fs.open_file().create(true).write(true).open(filename)?;

        Ok(writer)
    }
}
