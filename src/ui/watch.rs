use crate::core::config::Config;
use crate::core::error::{Error, Result};
use crate::indexing::discovery::is_notes_file;
use crate::indexing::parser::parse_markdown_file;
use crate::search::model::EmbeddingModel;
use crate::storage::state::{calculate_file_hash, get_file_modified_time, StateStore};
use crate::storage::vectors::{VectorEntry, VectorStore};
use notify_debouncer_full::{
    new_debouncer,
    notify::{RecursiveMode, Watcher},
    DebounceEventResult,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// File watcher for automatic indexing
pub struct FileWatcher {
    root_path: PathBuf,
    config: Config,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(root_path: &Path, config: Config) -> Result<Self> {
        Ok(Self {
            root_path: root_path.to_path_buf(),
            config,
        })
    }

    /// Start watching and processing file changes
    pub fn watch(&mut self) -> Result<()> {
        println!("Watching directory: {}", self.root_path.display());
        println!("Press Ctrl+C to stop watching...\n");

        let (tx, rx) = mpsc::channel();
        let root_path = self.root_path.clone();
        let config = self.config.clone();
        
        // Create debouncer with callback
        let mut debouncer = new_debouncer(
            Duration::from_secs(2),
            None,
            move |result: DebounceEventResult| {
                if let Ok(events) = result {
                    let _ = tx.send((events, root_path.clone(), config.clone()));
                }
            },
        )
        .map_err(|e| Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to create file watcher: {}", e),
        )))?;

        debouncer
            .watcher()
            .watch(&self.root_path, RecursiveMode::Recursive)
            .map_err(|e| Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to watch directory: {}", e),
            )))?;

        // Process events
        loop {
            match rx.recv() {
                Ok((events, root_path, config)) => {
                    Self::process_events_static(&events, &root_path, &config)?;
                }
                Err(_) => {
                    // Channel closed
                    break;
                }
            }
        }

        Ok(())
    }

    /// Process file change events (static version for use in closure)
    fn process_events_static(
        events: &[notify_debouncer_full::DebouncedEvent],
        root_path: &Path,
        config: &Config,
    ) -> Result<()> {
        let state_store = StateStore::open(config)?;
        let vector_store = VectorStore::open(config)?;
        
        // Initialize model once for all files in this batch
        // This avoids expensive re-initialization on every file change
        let model = match EmbeddingModel::init_verbose(config) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("⚠ Warning: Failed to initialize embedding model: {}", e);
                eprintln!("  Skipping file indexing in this batch.");
                return Ok(());
            }
        };

        for event in events {
            // DebouncedEvent contains paths (plural) - iterate through them
            for path in &event.paths {
                // Only process supported notes files
                if !is_notes_file(path) {
                    continue;
                }

                // Check if file exists (might have been deleted)
                if !path.exists() {
                    // File was deleted - remove from index
                    if let Ok(relative_path) = path.strip_prefix(root_path) {
                        let file_path_str = match relative_path.to_str() {
                            Some(s) => s,
                            None => {
                                eprintln!("⚠ Warning: Skipping deleted file with invalid UTF-8 path: {}", relative_path.display());
                                continue;
                            }
                        };
                        if let Err(e) = vector_store.remove_file(file_path_str) {
                            eprintln!("⚠ Warning: Failed to remove deleted file from index ({}): {}", relative_path.display(), e);
                        }
                        if let Err(e) = state_store.remove_file(file_path_str) {
                            eprintln!("⚠ Warning: Failed to remove deleted file from state ({}): {}", relative_path.display(), e);
                        }
                        println!("  ✗ Removed deleted file: {}", relative_path.display());
                    }
                    continue;
                }

                // Process file
                match path.strip_prefix(root_path) {
                    Ok(relative_path) => {
                        let file_path_str = match relative_path.to_str() {
                            Some(s) => s,
                            None => {
                                eprintln!("⚠ Warning: Skipping file with invalid UTF-8 path: {}", relative_path.display());
                                continue;
                            }
                        };
                        
                        // Check if file has changed
                        match (get_file_modified_time(path), calculate_file_hash(path)) {
                            (Ok(modified_time), Ok(hash)) => {
                                if let Ok(false) = state_store.has_file_changed(
                                    file_path_str,
                                    modified_time,
                                    &hash,
                                ) {
                                    // File hasn't changed, skip
                                    continue;
                                }

                                // Index the file
                                match Self::index_file_static(path, file_path_str, &state_store, &vector_store, &model) {
                                    Ok(_) => {
                                        // Update state
                                        if let Err(e) = state_store.update_file_state(
                                            file_path_str,
                                            modified_time,
                                            hash,
                                        ) {
                                            eprintln!("  ⚠ Warning: Failed to update state: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("  ✗ Failed to index {}: {}", relative_path.display(), e);
                                    }
                                }
                            }
                            (Err(e), _) => {
                                eprintln!("  ⚠ Warning: Could not get modification time for {}: {}", relative_path.display(), e);
                            }
                            (_, Err(e)) => {
                                eprintln!("  ⚠ Warning: Could not calculate hash for {}: {}", relative_path.display(), e);
                            }
                        }
                    }
                    Err(_) => {
                        // File is outside root path, skip
                        continue;
                    }
                }
            }
        }

        Ok(())
    }

    /// Index a single file (static version for use in closure)
    fn index_file_static(
        path: &Path,
        file_path_str: &str,
        _state_store: &StateStore,
        vector_store: &VectorStore,
        model: &EmbeddingModel,
    ) -> Result<()> {
        // Remove old vectors
        let _ = vector_store.remove_file(file_path_str);

        // Parse file
        let doc = parse_markdown_file(path)?;

        // Process chunks (model is already initialized and passed in)
        let chunks_to_embed: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
        // Use embed_passages for BGE model compatibility (better search quality)
        let embeddings = model.embed_passages(&chunks_to_embed)?;

        // Store vectors - pre-allocate entries for better performance
        let mut entries_to_insert = Vec::with_capacity(doc.chunks.len());
        for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
            entries_to_insert.push(VectorEntry::new(
                file_path_str.to_string(),
                chunk.chunk_index,
                embedding.clone(),
                chunk.text.clone(),
                chunk.context.clone(),
                chunk.start_line,
                chunk.end_line,
            ));
        }

        // Insert all entries
        for (i, entry) in entries_to_insert.iter().enumerate() {
            if let Err(e) = vector_store.insert(entry) {
                eprintln!("  ⚠ Warning: Failed to store vector for chunk {}: {}", i, e);
            }
        }

        println!("  ✓ Indexed: {} ({} chunks)", file_path_str, doc.chunks.len());
        Ok(())
    }
}

