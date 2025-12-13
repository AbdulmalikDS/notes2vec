use clap::Parser;
use notes2vec::{Cli, Config, discover_files, Error, Result};
use notes2vec::{EmbeddingModel, StateStore, calculate_file_hash, get_file_modified_time};
use notes2vec::{VectorStore, SearchTui, FileWatcher};
use std::path::PathBuf;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        notes2vec::ui::cli::Commands::Init { base_dir } => {
            handle_init(base_dir.as_deref())
        }
        notes2vec::ui::cli::Commands::Index { path, force, base_dir } => {
            handle_index(path.as_str(), *force, base_dir.as_deref())
        }
        notes2vec::ui::cli::Commands::Watch { path, base_dir } => {
            handle_watch(path.as_str(), base_dir.as_deref())
        }
        notes2vec::ui::cli::Commands::Search {
            query,
            limit,
            base_dir,
            interactive,
        } => {
            handle_search(query.as_deref(), *limit, base_dir.as_deref(), *interactive)
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

fn handle_index(path: &str, force: bool, base_dir: Option<&str>) -> Result<()> {
    println!("Indexing notes from: {}", path);
    
    // Check if initialized
    let base_path = base_dir.map(PathBuf::from);
    let config = Config::new(base_path)?;
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
    
    // Initialize embedding model once for all files
    println!("Initializing embedding model...");
    let model = match EmbeddingModel::init(&config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("⚠ Warning: Failed to initialize embedding model: {}. Using hash-based embeddings.", e);
            return Err(Error::Model(format!("Failed to initialize model: {}", e)));
        }
    };
    
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
        
        match notes2vec::indexing::parser::parse_markdown_file(&file.path) {
            Ok(doc) => {
                // Remove old vectors for this file if re-indexing
                if force {
                    if let Err(e) = vector_store.remove_file(file_path_str) {
                        eprintln!("  ⚠ Warning: Failed to remove old vectors for {}: {}", 
                                 file.relative_path.display(), e);
                    }
                }
                
                // Generate embeddings for all chunks
                let chunk_texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
                let embeddings = match model.embed(&chunk_texts) {
                    Ok(emb) => emb,
                    Err(e) => {
                        eprintln!("  ⚠ Warning: Failed to generate embeddings: {}. Skipping file.", e);
                        continue;
                    }
                };
                
                // Store vectors with embeddings
                for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
                    let vector_entry = notes2vec::VectorEntry::new(
                        file_path_str.to_string(),
                        chunk.chunk_index,
                        embedding.clone(),
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
    
    Ok(())
}

fn handle_watch(path: &str, base_dir: Option<&str>) -> Result<()> {
    // Check if initialized
    let base_path = base_dir.map(PathBuf::from);
    let config = Config::new(base_path)?;
    if !config.is_initialized() {
        return Err(Error::Config(
            "notes2vec is not initialized. Run 'notes2vec init' first.".to_string(),
        ));
    }
    
    let watch_path = PathBuf::from(path);
    if !watch_path.exists() {
        return Err(Error::Config(format!(
            "Path does not exist: {}",
            path
        )));
    }
    
    if !watch_path.is_dir() {
        return Err(Error::Config(format!(
            "Path is not a directory: {}",
            path
        )));
    }
    
    // Create watcher
    let mut watcher = FileWatcher::new(&watch_path, config)?;
    
    // Start watching (blocks until interrupted)
    watcher.watch()
}

fn handle_search(
    query: Option<&str>,
    limit: usize,
    base_dir: Option<&str>,
    interactive: bool,
) -> Result<()> {
    // Check if initialized
    let base_path = base_dir.map(PathBuf::from);
    let config = Config::new(base_path)?;
    if !config.is_initialized() {
        return Err(Error::Config(
            "notes2vec is not initialized. Run 'notes2vec init' first.".to_string(),
        ));
    }

    // Use interactive TUI mode if requested or no query provided
    if interactive || query.is_none() || query.unwrap().is_empty() {
        let mut tui = SearchTui::new(config)?;
        return tui.run();
    }

    // Non-interactive mode
    let query = query.unwrap();
    println!("Searching for: \"{}\"", query);

    // Open vector store
    let vector_store = VectorStore::open(&config)?;

    // Initialize embedding model and generate query embedding
    let model = EmbeddingModel::init(&config)?;
    let query_texts = vec![query.to_string()];
    let query_embeddings = model.embed(&query_texts)?;

    if query_embeddings.is_empty() {
        return Err(Error::Model("Failed to generate query embedding".to_string()));
    }

    let query_embedding = &query_embeddings[0];

    // Search for similar vectors
    let results = vector_store.search(&query_embedding, limit)?;

    if results.is_empty() {
        println!("\nNo results found.");
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
    }

    Ok(())
}

