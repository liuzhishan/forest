use super::task::Task;

/// Read data from local.
pub struct LocalReader {}

impl LocalReader {
    pub fn new() -> Self {
        Self {}
    }
}

impl Task for LocalReader {}
