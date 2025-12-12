use clap::Parser;
use notes2vec::cli::Cli;
use notes2vec::config::Config;
use notes2vec::discovery::discover_files;
use notes2vec::error::{Error, Result};
use notes2vec::state::{calculate_file_hash, get_file_modified_time, StateStore};
use notes2vec::vectors::VectorStore;
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
    
    // Open state store and vector store
    let state_store = StateStore::open(&config)?;
    let vector_store = VectorStore::open(&config)?;
    
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
    let mut chunks_indexed = 0;
    
    for file in &files {
        let file_path_str = file.relative_path.to_str().unwrap_or("");
        
        // Check if file has changed (unless force is true)
        if !force {
            match (get_file_modified_time(&file.path), calculate_file_hash(&file.path)) {
                (Ok(modified_time), Ok(hash)) => {
                    if let Ok(false) = state_store.has_file_changed(
                        file_path_str,
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
                // Remove old vectors for this file if re-indexing
                if force {
                    if let Err(e) = vector_store.remove_file(file_path_str) {
                        eprintln!("  ⚠ Warning: Failed to remove old vectors for {}: {}", 
                                 file.relative_path.display(), e);
                    }
                }
                
                // Process chunks (for now, just store metadata - embeddings will be added later)
                // TODO: Generate embeddings once model loading is implemented
                for chunk in &doc.chunks {
                    // Create a placeholder embedding (zeros) - will be replaced with actual embeddings
                    let embedding_dim = 768; // Default for nomic-embed-text-v1.5
                    let placeholder_embedding = vec![0.0f32; embedding_dim];
                    
                    let vector_entry = notes2vec::vectors::VectorEntry::new(
                        file_path_str.to_string(),
                        chunk.chunk_index,
                        placeholder_embedding,
                        chunk.text.clone(),
                        chunk.context.clone(),
                        chunk.start_line,
                        chunk.end_line,
                    );
                    
                    if let Err(e) = vector_store.insert(&vector_entry) {
                        eprintln!("  ⚠ Warning: Failed to store vector for chunk {}: {}", 
                                 chunk.chunk_index, e);
                    } else {
                        chunks_indexed += 1;
                    }
                }
                
                // Update state store
                if let (Ok(modified_time), Ok(hash)) =
                    (get_file_modified_time(&file.path), calculate_file_hash(&file.path))
                {
                    if let Err(e) = state_store.update_file_state(
                        file_path_str,
                        modified_time,
                        hash,
                    ) {
                        eprintln!("  ⚠ Warning: Failed to update state for {}: {}", 
                                 file.relative_path.display(), e);
                    }
                }
                
                println!("  ✓ {} ({} chunks)", file.relative_path.display(), doc.chunks.len());
                processed += 1;
            }
            Err(e) => {
                eprintln!("  ✗ {}: {}", file.relative_path.display(), e);
                errors += 1;
            }
        }
    }
    
    println!("\nIndexing complete!");
    println!("  Processed: {} files", processed);
    println!("  Chunks indexed: {}", chunks_indexed);
    if skipped > 0 {
        println!("  Skipped (unchanged): {} files", skipped);
    }
    if errors > 0 {
        println!("  Errors: {} files", errors);
    }
    println!("\nNote: Embedding generation will be implemented next. Currently storing placeholder vectors.");
    
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
    
    // Open vector store
    let vector_store = VectorStore::open(&config)?;
    
    // TODO: Generate query embedding once model is implemented
    // For now, use a placeholder embedding
    let embedding_dim = 768;
    let query_embedding = vec![0.0f32; embedding_dim];
    
    // Search for similar vectors
    match vector_store.search(&query_embedding, limit) {
        Ok(results) => {
            if results.is_empty() {
                println!("\nNo results found.");
                println!("Note: Embedding generation is not yet implemented. Search will work once embeddings are generated.");
            } else {
                println!("\nFound {} results:", results.len());
                for (i, (entry, similarity)) in results.iter().enumerate() {
                    println!("\n{}. {} (similarity: {:.3})", i + 1, entry.file_path, similarity);
                    if !entry.context.is_empty() {
                        println!("   Context: {}", entry.context);
                    }
                    // Show preview of text (first 150 chars)
                    let preview: String = entry.text.chars().take(150).collect();
                    println!("   Preview: {}...", preview);
                    println!("   Lines: {}-{}", entry.start_line, entry.end_line);
                }
                println!("\nNote: Embedding generation is not yet implemented. Similarity scores are placeholders.");
            }
        }
        Err(e) => {
            return Err(Error::Database(format!("Search failed: {}", e)));
        }
    }
    
    Ok(())
}

