use notes2vec::config::Config;
use notes2vec::error::Result;

fn main() -> Result<()> {
    println!("notes2vec - Local semantic search for personal notes");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    
    // For now, just test config initialization
    let config = Config::new(None)?;
    println!("Configuration directory: {:?}", config.base_dir);
    
    Ok(())
}

