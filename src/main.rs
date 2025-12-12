use clap::Parser;
use notes2vec::cli::Cli;
use notes2vec::config::Config;
use notes2vec::discovery::discover_files;
use notes2vec::error::{Error, Result};
use notes2vec::state::{calculate_file_hash, get_file_modified_time, StateStore};
use std::path::PathBuf;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        notes2vec::cli::Commands::Init { base_dir } => {
            handle_init(base_dir.as_deref())
        }
        notes2vec::cli::Commands::Index { path, force } => handle_index(path.as_str(), *force),
        notes2vec::cli::Commands::Watch { path } => handle_watch(path.as_str()),
        notes2vec::cli::Commands::Search { query, limit } => handle_search(query.as_str(), *limit),
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
    
    // Open state store
    let state_store = StateStore::open(&config)?;
    
    let root_path = PathBuf::from(path);
    
    // Discover all Markdown files
    println!("Discovering Markdown files...");
    let files = discover_files(&root_path)?;
    println!("Found {} Markdown files", files.len());
    
    if files.is_empty() {
        println!("No Markdown files found in {}", path);
        return Ok(());
    }
    
    // Process files
    println!("Processing files...");
    let mut processed = 0;
    let mut skipped = 0;
    let mut errors = 0;
    
    for file in &files {
        // Check if file has changed (unless force is true)
        if !force {
            match (get_file_modified_time(&file.path), calculate_file_hash(&file.path)) {
                (Ok(modified_time), Ok(hash)) => {
                    if let Ok(false) = state_store.has_file_changed(
                        file.relative_path.to_str().unwrap_or(""),
                        modified_time,
                        &hash,
                    ) {
                        skipped += 1;
                        continue;
                    }
                }
                _ => {
                    // If we can't get file info, process it anyway
                }
            }
        }
        
        match notes2vec::parser::parse_markdown_file(&file.path) {
            Ok(doc) => {
                // Update state store
                if let (Ok(modified_time), Ok(hash)) =
                    (get_file_modified_time(&file.path), calculate_file_hash(&file.path))
                {
                    if let Err(e) = state_store.update_file_state(
                        file.relative_path.to_str().unwrap_or(""),
                        modified_time,
                        hash,
                    ) {
                        eprintln!("  ⚠ Warning: Failed to update state for {}: {}", 
                                 file.relative_path.display(), e);
                    }
                }
                
                println!("  ✓ {} ({} chunks)", file.relative_path.display(), doc.chunks.len());
                processed += 1;
                // TODO: Generate embeddings and store in database
            }
            Err(e) => {
                eprintln!("  ✗ {}: {}", file.relative_path.display(), e);
                errors += 1;
            }
        }
    }
    
    println!("\nIndexing complete!");
    println!("  Processed: {} files", processed);
    if skipped > 0 {
        println!("  Skipped (unchanged): {} files", skipped);
    }
    if errors > 0 {
        println!("  Errors: {} files", errors);
    }
    println!("\nNote: Embedding generation and database storage will be implemented next.");
    
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

