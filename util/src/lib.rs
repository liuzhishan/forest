use env_logger;
use log::info;

pub fn init_log() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
}
