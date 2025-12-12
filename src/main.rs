use notes2vec::cli::Cli;
use notes2vec::config::Config;
use notes2vec::error::{Error, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        notes2vec::cli::Commands::Init { base_dir } => {
            handle_init(base_dir.as_deref())
        }
        notes2vec::cli::Commands::Index { path, force } => {
            handle_index(path, *force)
        }
        notes2vec::cli::Commands::Watch { path } => {
            handle_watch(path)
        }
        notes2vec::cli::Commands::Search { query, limit } => {
            handle_search(query, *limit)
        }
    }
}

fn handle_init(base_dir: Option<&str>) -> Result<()> {
    println!("Initializing notes2vec...");
    
    let base_path = base_dir
        .map(PathBuf::from)
        .or_else(|| Config::default_base_dir().ok());
    
    let config = Config::new(base_path)?;
    
    if config.is_initialized() {
        println!("notes2vec is already initialized at: {:?}", config.base_dir);
        println!("To reinitialize, delete the directory and run 'init' again.");
        return Ok(());
    }
    
    config.init()?;
    println!("✓ Created configuration directory: {:?}", config.base_dir);
    println!("✓ Created database directory: {:?}", config.database_dir);
    println!("✓ Created models directory: {:?}", config.models_dir);
    println!("✓ Created state directory: {:?}", config.state_path.parent().unwrap());
    
    println!("\nInitialization complete!");
    println!("Next steps:");
    println!("  1. Index your notes: notes2vec index /path/to/notes");
    println!("  2. Or watch for changes: notes2vec watch /path/to/notes");
    
    Ok(())
}

fn handle_index(path: &str, force: bool) -> Result<()> {
    println!("Indexing notes from: {}", path);
    
    // Check if initialized
    let config = Config::new(None)?;
    if !config.is_initialized() {
        return Err(Error::Config(
            "notes2vec is not initialized. Run 'notes2vec init' first.".to_string(),
        ));
    }
    
    // TODO: Implement indexing logic
    println!("Indexing functionality will be implemented next...");
    println!("Path: {}", path);
    println!("Force: {}", force);
    
    Ok(())
}

fn handle_watch(path: &str) -> Result<()> {
    println!("Watching directory for changes: {}", path);
    
    // Check if initialized
    let config = Config::new(None)?;
    if !config.is_initialized() {
        return Err(Error::Config(
            "notes2vec is not initialized. Run 'notes2vec init' first.".to_string(),
        ));
    }
    
    // TODO: Implement file watching logic
    println!("File watching functionality will be implemented next...");
    println!("Path: {}", path);
    
    Ok(())
}

fn handle_search(query: &str, limit: usize) -> Result<()> {
    println!("Searching for: \"{}\"", query);
    
    // Check if initialized
    let config = Config::new(None)?;
    if !config.is_initialized() {
        return Err(Error::Config(
            "notes2vec is not initialized. Run 'notes2vec init' first.".to_string(),
        ));
    }
    
    // TODO: Implement search logic
    println!("Search functionality will be implemented next...");
    println!("Query: {}", query);
    println!("Limit: {}", limit);
    
    Ok(())
}

