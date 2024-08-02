use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    util::init_log();

    Ok(())
}
