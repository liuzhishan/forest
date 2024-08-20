use std::default;

/// Status code.
#[derive(Default, Clone)]
pub enum StatusCode {
    #[default]
    Ok,
    NotFound,
    Corruption,
    NotSupported,
    InvalidArgument,
    IOError,
    ShutdownInProgress,
    Timeout,
    Aborted,
    Busy,
    Expired,
    Duplicate,
    Compacted,
    EndOfFile,
    None,
}

/// Status of different task.
#[derive(Default, Clone)]
pub struct Status {
    pub code: StatusCode,
    pub message: String,
}
