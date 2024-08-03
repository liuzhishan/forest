/// Worker.
pub trait Worker {
}

/// Read data from hdfs.
pub struct HdfsReader {
}

impl HdfsReader {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Worker for HdfsReader {

}

/// Assembly the data into batch.
///
/// The batch size must be provided in construction of BatchBuilder.
pub struct BatchBuilder {

}
