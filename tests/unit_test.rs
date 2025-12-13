use notes2vec::{Config, Result};
use notes2vec::{VectorEntry, VectorStore};
use notes2vec::{StateStore, calculate_file_hash, get_file_modified_time};
use notes2vec::discover_files;
use notes2vec::indexing::parser::parse_markdown;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Test VectorEntry serialization and deserialization
#[test]
fn test_vector_entry_roundtrip() -> Result<()> {
    let entry = VectorEntry::new(
        "test.md".to_string(),
        5,
        vec![0.1, 0.2, 0.3, 0.4],
        "Test content".to_string(),
        "Section > Subsection".to_string(),
        10,
        20,
    );

    let json = entry.to_json()?;
    let deserialized = VectorEntry::from_json(&json)?;
    
    assert_eq!(deserialized.file_path, entry.file_path);
    assert_eq!(deserialized.chunk_index, entry.chunk_index);
    assert_eq!(deserialized.embedding, entry.embedding);
    assert_eq!(deserialized.text, entry.text);
    assert_eq!(deserialized.context, entry.context);
    
    Ok(())
}

/// Test VectorStore operations
#[test]
fn test_vector_store_operations() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let config = Config::new(Some(base_dir))?;
    config.init()?;

    let store = VectorStore::open(&config)?;

    // Insert multiple entries
    for i in 0..10 {
        let entry = VectorEntry::new(
            format!("file{}.md", i % 3), // 3 files, 10 chunks total
            i,
            vec![0.1 * i as f32, 0.2, 0.3],
            format!("Chunk {}", i),
            "Context".to_string(),
            1,
            10,
        );
        store.insert(&entry)?;
    }

    // Test get_file_count
    assert_eq!(store.get_file_count()?, 3);

    // Test get_file_vectors
    let vectors = store.get_file_vectors("file0.md")?;
    assert!(vectors.len() >= 3); // At least 3 chunks for file0

    // Test search
    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 5)?;
    assert!(results.len() <= 5);
    assert!(!results.is_empty());

    Ok(())
}

/// Test StateStore file change detection
#[test]
fn test_state_store_change_detection() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let config = Config::new(Some(base_dir))?;
    config.init()?;

    let store = StateStore::open(&config)?;
    let test_file = temp_dir.path().join("test.md");

    // Create file
    fs::write(&test_file, "Initial content")?;
    let modified_time = get_file_modified_time(&test_file)?;
    let hash = calculate_file_hash(&test_file)?;

    // New file should be considered changed
    assert!(store.has_file_changed("test.md", modified_time, &hash)?);

    // Update state
    store.update_file_state("test.md", modified_time, hash.clone())?;

    // Same file should not be changed
    assert!(!store.has_file_changed("test.md", modified_time, &hash)?);

    // Modify file
    std::thread::sleep(std::time::Duration::from_millis(100));
    fs::write(&test_file, "Modified content")?;
    let new_modified_time = get_file_modified_time(&test_file)?;
    let new_hash = calculate_file_hash(&test_file)?;

    // Modified file should be detected as changed
    assert!(store.has_file_changed("test.md", new_modified_time, &new_hash)?);

    Ok(())
}

/// Test markdown parsing edge cases
#[test]
fn test_markdown_parsing_edge_cases() -> Result<()> {
    // Test empty file
    let doc = parse_markdown("", Path::new("empty.md"))?;
    assert_eq!(doc.title, "empty");

    // Test file without headers
    let doc = parse_markdown("Just some text without headers.", Path::new("no_headers.md"))?;
    assert_eq!(doc.title, "no_headers");

    // Test multiple H1 headers (first one should be title)
    let content = r#"# First Title

Content.

# Second Title

More content.
"#;
    let doc = parse_markdown(content, Path::new("test.md"))?;
    assert_eq!(doc.title, "First Title");

    // Test frontmatter with title override
    let content = r#"---
title: Frontmatter Title
---

# Markdown Title

Content.
"#;
    let doc = parse_markdown(content, Path::new("test.md"))?;
    assert_eq!(doc.metadata.title, Some("Frontmatter Title".to_string()));
    assert_eq!(doc.title, "Markdown Title"); // Title from H1, not frontmatter

    Ok(())
}

/// Test file discovery edge cases
#[test]
fn test_file_discovery_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let test_dir = temp_dir.path().join("notes");
    fs::create_dir_all(&test_dir)?;

    // Test empty directory
    let files = discover_files(&test_dir)?;
    assert_eq!(files.len(), 0);

    // Test with only non-note files (note: .txt files ARE supported, so use other extensions)
    fs::write(test_dir.join("file.js"), "Not a note file")?;
    fs::write(test_dir.join("file.py"), "Not a note file")?;
    fs::write(test_dir.join("file.json"), "Not a note file")?;
    let files = discover_files(&test_dir)?;
    assert_eq!(files.len(), 0);

    // Test with mixed files
    fs::write(test_dir.join("file.md"), "# Markdown")?;
    let files = discover_files(&test_dir)?;
    assert_eq!(files.len(), 1);

    Ok(())
}

/// Test file hash calculation
#[test]
fn test_file_hash() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.txt");

    // Same content should produce same hash
    fs::write(&test_file, "Test content")?;
    let hash1 = calculate_file_hash(&test_file)?;
    let hash2 = calculate_file_hash(&test_file)?;
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 64); // SHA256 produces 64 hex chars

    // Different content should produce different hash
    fs::write(&test_file, "Different content")?;
    let hash3 = calculate_file_hash(&test_file)?;
    assert_ne!(hash1, hash3);

    Ok(())
}

/// Test configuration initialization
#[test]
fn test_config_initialization() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");

    let config = Config::new(Some(base_dir.clone()))?;
    assert!(!config.is_initialized());

    config.init()?;
    assert!(config.is_initialized());
    assert!(config.base_dir.exists());
    assert!(config.database_dir.exists());
    assert!(config.models_dir.exists());

    Ok(())
}

/// Test vector store search with empty database
#[test]
fn test_vector_store_search_empty() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let config = Config::new(Some(base_dir))?;
    config.init()?;

    let store = VectorStore::open(&config)?;
    let query = vec![1.0, 0.0, 0.0];
    let results = store.search(&query, 10)?;
    assert_eq!(results.len(), 0);

    Ok(())
}

/// Test vector store remove file that doesn't exist
#[test]
fn test_vector_store_remove_nonexistent() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let config = Config::new(Some(base_dir))?;
    config.init()?;

    let store = VectorStore::open(&config)?;
    let removed = store.remove_file("nonexistent.md")?;
    assert_eq!(removed, 0);

    Ok(())
}

/// Test state store remove file that doesn't exist
#[test]
fn test_state_store_remove_nonexistent() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let config = Config::new(Some(base_dir))?;
    config.init()?;

    let store = StateStore::open(&config)?;
    // Should not error when removing non-existent file
    store.remove_file("nonexistent.md")?;

    Ok(())
}

