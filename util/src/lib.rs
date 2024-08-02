use log::info;
use env_logger;

pub fn init_log() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
}
