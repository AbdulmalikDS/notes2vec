use notes2vec::{Config, discover_files, Result};
use notes2vec::EmbeddingModel;
use notes2vec::indexing::parser::parse_markdown_file;
use notes2vec::{StateStore, calculate_file_hash, get_file_modified_time};
use notes2vec::VectorStore;
use std::fs;
use tempfile::TempDir;

/// End-to-end functionality test: full indexing and search workflow
#[test]
fn test_end_to_end_workflow() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    // 1. Initialize
    let config = Config::new(Some(base_dir.clone()))?;
    config.init()?;
    assert!(config.is_initialized());
    
    // 2. Create test notes
    fs::create_dir_all(&notes_dir)?;
    fs::write(
        notes_dir.join("rust.md"),
        r#"# Rust Programming

Rust is a systems programming language focused on safety and performance.

## Ownership

Rust's ownership system ensures memory safety without garbage collection.

## Borrowing

You can borrow references to data without taking ownership.
"#,
    )?;
    
    fs::write(
        notes_dir.join("database.md"),
        r#"# Database Configuration

PostgreSQL is a powerful relational database system.

## Setup

To set up PostgreSQL, you need to configure the connection settings.

## Performance

Indexing and query optimization are crucial for database performance.
"#,
    )?;
    
    // 3. Discover files
    let files = discover_files(&notes_dir)?;
    assert_eq!(files.len(), 2);
    
    // 4. Open stores
    let state_store = StateStore::open(&config)?;
    let vector_store = VectorStore::open(&config)?;
    let model = EmbeddingModel::init(&config)?;
    
    // 5. Index files
    for file in &files {
        let doc = parse_markdown_file(&file.path)?;
        assert!(!doc.chunks.is_empty());
        
        // Generate embeddings
        let texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = model.embed(&texts)?;
        assert_eq!(embeddings.len(), doc.chunks.len());
        
        // Store vectors
        let file_path_str = file.relative_path.to_str().unwrap();
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
            vector_store.insert(&vector_entry)?;
        }
        
        // Update state
        let modified_time = get_file_modified_time(&file.path)?;
        let hash = calculate_file_hash(&file.path)?;
        state_store.update_file_state(file_path_str, modified_time, hash)?;
    }
    
    // 6. Test search functionality
    let queries = vec![
        "programming language",
        "database setup",
        "memory safety",
    ];
    
    for query in queries {
        let query_texts = vec![query.to_string()];
        let query_embeddings = model.embed(&query_texts)?;
        let results = vector_store.search(&query_embeddings[0], 5)?;
        
        assert!(!results.is_empty(), "No results for query: {}", query);
        
        // Verify results are sorted by similarity (descending)
        for i in 0..results.len().saturating_sub(1) {
            assert!(
                results[i].1 >= results[i + 1].1,
                "Results not sorted by similarity"
            );
        }
    }
    
    // 7. Test file removal
    let rust_vectors_before = vector_store.get_file_vectors("rust.md")?;
    assert!(!rust_vectors_before.is_empty());
    
    vector_store.remove_file("rust.md")?;
    
    let rust_vectors_after = vector_store.get_file_vectors("rust.md")?;
    assert!(rust_vectors_after.is_empty());
    
    Ok(())
}

/// Test semantic search quality
#[test]
fn test_semantic_search_quality() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    let vector_store = VectorStore::open(&config)?;
    let model = EmbeddingModel::init(&config)?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create documents with related but different wording
    fs::write(
        notes_dir.join("doc1.md"),
        "# How to install software\n\nDownload the installer and run it.",
    )?;
    
    fs::write(
        notes_dir.join("doc2.md"),
        "# Setting up applications\n\nGet the setup file and execute it.",
    )?;
    
    fs::write(
        notes_dir.join("doc3.md"),
        "# Cooking recipes\n\nMix ingredients and bake.",
    )?;
    
    // Index all documents
    let files = discover_files(&notes_dir)?;
    for file in &files {
        let doc = parse_markdown_file(&file.path)?;
        let texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = model.embed(&texts)?;
        
        let file_path_str = file.relative_path.to_str().unwrap();
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
            vector_store.insert(&vector_entry)?;
        }
    }
    
    // Search with semantically similar query
    let query_texts = vec!["installing programs".to_string()];
    let query_embeddings = model.embed(&query_texts)?;
    let results = vector_store.search(&query_embeddings[0], 3)?;
    
    // Should find doc1 and doc2 (both about installation) but not doc3 (cooking)
    assert!(!results.is_empty());
    
    let found_files: Vec<&str> = results.iter().map(|(e, _)| e.file_path.as_str()).collect();
    
    // At least one installation-related doc should be in top results
    let has_installation_doc = found_files.iter().any(|f| f.contains("doc1") || f.contains("doc2"));
    assert!(has_installation_doc, "Semantic search should find installation-related docs");
    
    Ok(())
}

/// Test chunking with large documents
#[test]
fn test_large_document_chunking() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("large.md");
    
    // Create a large document
    let mut content = String::from("# Large Document\n\n");
    for i in 1..=50 {
        content.push_str(&format!("## Section {}\n\n", i));
        content.push_str(&format!("This is paragraph one in section {}. ", i));
        content.push_str(&format!("This is paragraph two in section {}. ", i));
        content.push_str(&format!("This is paragraph three in section {}. ", i));
        content.push_str("\n\n");
    }
    
    fs::write(&test_file, content)?;
    
    let doc = parse_markdown_file(&test_file)?;
    
    // Should have multiple chunks
    assert!(doc.chunks.len() > 1, "Large document should be chunked");
    
    // Each chunk should have reasonable size
    for chunk in &doc.chunks {
        assert!(!chunk.text.is_empty());
        assert!(chunk.text.len() <= 1000, "Chunks should not be too large");
    }
    
    Ok(())
}

/// Test error handling
#[test]
fn test_error_handling() {
    // Test non-existent directory
    let result = discover_files(std::path::Path::new("/nonexistent/path"));
    assert!(result.is_err());
    
    // Test non-existent file parsing
    let result = parse_markdown_file(std::path::Path::new("/nonexistent/file.md"));
    assert!(result.is_err());
}

/// Test configuration edge cases
#[test]
fn test_config_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_config");
    
    // Test with custom base dir
    let config = Config::new(Some(base_dir.clone()))?;
    assert_eq!(config.base_dir, base_dir);
    
    // Test default base dir
    let config_default = Config::new(None)?;
    assert!(config_default.base_dir.to_string_lossy().contains(".notes2vec"));
    
    Ok(())
}

