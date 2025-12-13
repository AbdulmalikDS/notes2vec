use notes2vec::{Config, discover_files, Result};
use notes2vec::EmbeddingModel;
use notes2vec::indexing::parser::parse_markdown_file;
use notes2vec::{StateStore, calculate_file_hash, get_file_modified_time};
use notes2vec::VectorStore;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_real_markdown_file() -> Result<()> {
    // Create temporary directory
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    // Setup
    let config = Config::new(Some(base_dir.clone()))?;
    config.init()?;
    let state_store = StateStore::open(&config)?;
    let vector_store = VectorStore::open(&config)?;
    
    // Create test notes directory
    fs::create_dir_all(&notes_dir)?;
    
    // Create a real markdown file
    let test_content = r#"# Test Document

This is a test document for notes2vec.

## Section 1

This is the first section with some content about testing.

## Section 2

This is the second section with more content.
"#;
    
    let test_file = notes_dir.join("test.md");
    fs::write(&test_file, test_content)?;
    
    // Test discovery
    let files = discover_files(&notes_dir)?;
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, test_file);
    
    // Test parsing
    let doc = parse_markdown_file(&test_file)?;
    assert_eq!(doc.title, "Test Document");
    assert!(!doc.chunks.is_empty());
    assert!(doc.chunks.len() >= 2); // At least 2 sections
    
    // Test state tracking
    let modified_time = get_file_modified_time(&test_file)?;
    let hash = calculate_file_hash(&test_file)?;
    
    assert!(state_store.has_file_changed("test.md", modified_time, &hash)?);
    
    state_store.update_file_state("test.md", modified_time, hash.clone())?;
    assert!(!state_store.has_file_changed("test.md", modified_time, &hash)?);
    
    // Test embedding generation
    let model = EmbeddingModel::init(&config)?;
    let embedding_dim = model.embedding_dim();
    assert_eq!(embedding_dim, 768);
    
    let texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = model.embed(&texts)?;
    assert_eq!(embeddings.len(), doc.chunks.len());
    assert_eq!(embeddings[0].len(), embedding_dim);
    
    // Test vector storage
    for (_i, (chunk, embedding)) in doc.chunks.iter().zip(embeddings.iter()).enumerate() {
        let vector_entry = notes2vec::vectors::VectorEntry::new(
            "test.md".to_string(),
            chunk.chunk_index,
            embedding.clone(),
            chunk.text.clone(),
            chunk.context.clone(),
            chunk.start_line,
            chunk.end_line,
        );
        
        vector_store.insert(&vector_entry)?;
        
        // Verify it was stored
        let chunk_id = format!("test.md:{}", chunk.chunk_index);
        let retrieved = vector_store.get(&chunk_id)?;
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.text, chunk.text);
        assert_eq!(retrieved_entry.embedding.len(), embedding_dim);
    }
    
    // Test search
    let query_texts = vec!["testing content".to_string()];
    let query_embeddings = model.embed(&query_texts)?;
    assert!(!query_embeddings.is_empty());
    
    let results = vector_store.search(&query_embeddings[0], 5)?;
    assert!(!results.is_empty());
    
    // Verify results contain our test file
    let has_test_file = results.iter().any(|(entry, _)| entry.file_path == "test.md");
    assert!(has_test_file);
    
    Ok(())
}

#[test]
fn test_multiple_files_indexing() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    let vector_store = VectorStore::open(&config)?;
    let model = EmbeddingModel::init(&config)?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create multiple test files
    let files = vec![
        ("file1.md", "# File 1\n\nContent about Rust programming."),
        ("file2.md", "# File 2\n\nContent about database configuration."),
        ("file3.md", "# File 3\n\nContent about API design."),
    ];
    
    for (filename, content) in &files {
        fs::write(notes_dir.join(filename), content)?;
    }
    
    // Discover and index
    let discovered = discover_files(&notes_dir)?;
    assert_eq!(discovered.len(), 3);
    
    // Index each file
    for file in &discovered {
        let doc = parse_markdown_file(&file.path)?;
        let texts: Vec<String> = doc.chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = model.embed(&texts)?;
        
        let file_path_str = file.relative_path.to_str().unwrap();
        
        for (chunk, embedding) in doc.chunks.iter().zip(embeddings.iter()) {
            let vector_entry = notes2vec::vectors::VectorEntry::new(
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
    
    // Test search across multiple files
    let query_texts = vec!["programming language".to_string()];
    let query_embeddings = model.embed(&query_texts)?;
    let results = vector_store.search(&query_embeddings[0], 10)?;
    
    // Should find results from multiple files
    assert!(!results.is_empty());
    
    // Verify we can get vectors for a specific file
    let file1_vectors = vector_store.get_file_vectors("file1.md")?;
    assert!(!file1_vectors.is_empty());
    
    Ok(())
}

