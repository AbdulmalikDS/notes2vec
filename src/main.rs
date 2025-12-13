use clap::Parser;
use notes2vec::{Cli, Config, discover_files, Error, Result};
use notes2vec::{EmbeddingModel, StateStore, calculate_file_hash, get_file_modified_time};
use notes2vec::{VectorStore, VectorEntry, SearchTui, FileWatcher};
use std::path::PathBuf;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(notes2vec::ui::cli::Commands::Init { base_dir }) => {
            handle_init(base_dir.as_deref())
        }
        Some(notes2vec::ui::cli::Commands::Index { path, force, base_dir }) => {
            handle_index(path.as_str(), *force, base_dir.as_deref())
        }
        Some(notes2vec::ui::cli::Commands::Watch { path, base_dir }) => {
            handle_watch(path.as_str(), base_dir.as_deref())
        }
        Some(notes2vec::ui::cli::Commands::Search {
            query,
            limit,
            base_dir,
            interactive,
        }) => {
            handle_search(query.as_deref(), *limit, base_dir.as_deref(), *interactive)
        }
        None => {
            // No subcommand provided - always open TUI for interactive search
            // If query is provided, it will be used as initial search, otherwise TUI starts empty
            handle_search(cli.query.as_deref(), cli.limit, cli.base_dir.as_deref(), true)
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
    println!("✓ Created state directory: {:?}", config.state_path.parent().unwrap_or(&config.base_dir));
    
    println!("\nInitialization complete!");
    println!("Next steps:");
    println!("  1. Index your notes: notes2vec index /path/to/notes");
    println!("  2. Or watch for changes: notes2vec watch /path/to/notes");
    
    Ok(())
}

fn handle_index(path: &str, force: bool, base_dir: Option<&str>) -> Result<()> {
    println!("Indexing notes from: {}", path);
    
    // Validate path exists and is a directory
    let root_path = PathBuf::from(path);
    if !root_path.exists() {
        return Err(Error::Config(format!(
            "Path does not exist: {}",
            path
        )));
    }
    if !root_path.is_dir() {
        return Err(Error::Config(format!(
            "Path is not a directory: {}",
            path
        )));
    }
    
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
    let model = match EmbeddingModel::init_verbose(&config) {
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
        // Convert path to string, skip if invalid UTF-8
        let file_path_str = match file.relative_path.to_str() {
            Some(s) => s,
            None => {
                eprintln!("  ⚠ Warning: Skipping file with invalid UTF-8 path: {}", file.relative_path.display());
                errors += 1;
                continue;
            }
        };
        
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
                (Err(e), _) => {
                    eprintln!("  ⚠ Warning: Could not get modification time for {}: {}. Processing anyway.", 
                             file.relative_path.display(), e);
                }
                (_, Err(e)) => {
                    eprintln!("  ⚠ Warning: Could not calculate hash for {}: {}. Processing anyway.", 
                             file.relative_path.display(), e);
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
                // Reuse chunk texts to avoid extra allocations
                // Use embed_passages for BGE model compatibility (better search quality)
                let chunk_texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
                let embeddings = match model.embed_passages(&chunk_texts) {
                    Ok(emb) => emb,
                    Err(e) => {
                        eprintln!("  ⚠ Warning: Failed to generate embeddings: {}. Skipping file.", e);
                        continue;
                    }
                };
                
                // Store vectors with embeddings - batch insert for better performance
                // Pre-allocate vector entries to reduce allocations
                let mut entries_to_insert = Vec::with_capacity(doc.chunks.len());
                for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
                    entries_to_insert.push(notes2vec::VectorEntry::new(
                        file_path_str.to_string(),
                        chunk.chunk_index,
                        embedding.clone(),
                        chunk.text.clone(),
                        chunk.context.clone(),
                        chunk.start_line,
                        chunk.end_line,
                    ));
                }
                
                // Insert all entries (redb handles transactions efficiently)
                for entry in &entries_to_insert {
                    if let Err(e) = vector_store.insert(entry) {
                        eprintln!("  ⚠ Warning: Failed to store vector for chunk {}: {}", 
                                 entry.chunk_index, e);
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
    if interactive || query.map(|q| q.is_empty()).unwrap_or(true) {
        let mut tui = SearchTui::new(config)?;
        return tui.run();
    }

    // Non-interactive mode
    let query = query.unwrap();
    println!("Searching for: \"{}\"", query);

    // Open vector store
    let vector_store = VectorStore::open(&config)?;

    // Initialize embedding model and generate query embedding
    let model = EmbeddingModel::init_verbose(&config)?;
    let query_texts = vec![query.to_string()];
    let query_embeddings = model.embed_queries(&query_texts)?;

    if query_embeddings.is_empty() {
        return Err(Error::Model("Failed to generate query embedding".to_string()));
    }

    let query_embedding = &query_embeddings[0];

    // Search for similar vectors (get more candidates for deduplication)
    let results = vector_store.search(&query_embedding, limit * 3)?;

    // Deduplicate: keep best match per file (like TUI does)
    // Optimized: Pre-allocate HashMap and avoid unnecessary clones
    use std::collections::HashMap;
    let mut best_by_file: HashMap<String, (VectorEntry, f32)> = HashMap::with_capacity(results.len());
    for (entry, sim) in results {
        match best_by_file.get_mut(&entry.file_path) {
            Some(current) => {
                if sim > current.1 {
                    *current = (entry, sim);
                }
            }
            None => {
                best_by_file.insert(entry.file_path.clone(), (entry, sim));
            }
        }
    }
    let mut deduped: Vec<(VectorEntry, f32)> = best_by_file.into_values().collect();
    deduped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    deduped.truncate(limit);

    if deduped.is_empty() {
        println!("\nNo results found.");
    } else {
        println!("\nFound {} results:", deduped.len());
        for (i, (entry, similarity)) in deduped.iter().enumerate() {
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

