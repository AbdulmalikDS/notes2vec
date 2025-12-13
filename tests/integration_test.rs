use notes2vec::{Config, discover_files, Result};
use notes2vec::indexing::parser::parse_markdown_file;
use notes2vec::{StateStore, calculate_file_hash, get_file_modified_time};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_init_command() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    
    let config = Config::new(Some(base_dir.clone()))?;
    
    // Should not be initialized yet
    assert!(!config.is_initialized());
    
    // Initialize
    config.init()?;
    
    // Should be initialized now
    assert!(config.is_initialized());
    
    // Check directories were created
    assert!(config.base_dir.exists());
    assert!(config.database_dir.exists());
    assert!(config.models_dir.exists());
    assert!(config.state_path.parent().unwrap().exists());
    
    Ok(())
}

#[test]
fn test_file_discovery() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let test_dir = temp_dir.path().join("notes");
    fs::create_dir_all(&test_dir)?;
    
    // Create some test markdown files
    fs::write(test_dir.join("file1.md"), "# Test File 1\n\nContent here.")?;
    fs::write(test_dir.join("file2.md"), "# Test File 2\n\nMore content.")?;
    fs::write(test_dir.join("file3.txt"), "Not a markdown file")?;
    fs::write(test_dir.join("file4.markdown"), "# Test File 4\n\nContent.")?;
    
    // Create a subdirectory
    let subdir = test_dir.join("subdir");
    fs::create_dir_all(&subdir)?;
    fs::write(subdir.join("file5.md"), "# Test File 5\n\nContent.")?;
    
    let files = discover_files(&test_dir)?;
    
    // Should find 4 markdown files (file1.md, file2.md, file4.markdown, file5.md)
    assert_eq!(files.len(), 4);
    
    // Verify all files are markdown
    for file in &files {
        assert!(file.is_markdown);
    }
    
    Ok(())
}

#[test]
fn test_markdown_parsing() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.md");
    
    let content = r#"---
title: Test Document
tags: [rust, testing]
---

# Main Title

This is the first paragraph.

## Section 1

This is content in section 1.

## Section 2

This is content in section 2.
"#;
    
    fs::write(&test_file, content)?;
    
    let doc = parse_markdown_file(&test_file)?;
    
    // Check title
    assert_eq!(doc.title, "Main Title");
    
    // Check metadata
    assert_eq!(doc.metadata.title, Some("Test Document".to_string()));
    assert_eq!(doc.metadata.tags.len(), 2);
    
    // Check chunks were created
    assert!(!doc.chunks.is_empty());
    
    // Check header hierarchy
    assert!(!doc.header_hierarchy.is_empty());
    
    Ok(())
}

#[test]
fn test_state_store() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    // Create state store
    let state_store = StateStore::open(&config)?;
    
    // Create a test file
    let test_file = temp_dir.path().join("test.md");
    fs::write(&test_file, "# Test\n\nContent")?;
    
    let modified_time = get_file_modified_time(&test_file)?;
    let hash = calculate_file_hash(&test_file)?;
    
    // File should not exist in state yet
    let state = state_store.get_file_state("test.md")?;
    assert!(state.is_none());
    
    // Check if file has changed (should be true for new file)
    assert!(state_store.has_file_changed("test.md", modified_time, &hash)?);
    
    // Update state
    state_store.update_file_state("test.md", modified_time, hash.clone())?;
    
    // File should exist now
    let state = state_store.get_file_state("test.md")?;
    assert!(state.is_some());
    
    // File should not have changed
    assert!(!state_store.has_file_changed("test.md", modified_time, &hash)?);
    
    // Modify the file
    fs::write(&test_file, "# Test\n\nModified content")?;
    let new_hash = calculate_file_hash(&test_file)?;
    
    // File should have changed
    assert!(state_store.has_file_changed("test.md", modified_time, &new_hash)?);
    
    Ok(())
}

#[test]
fn test_full_indexing_workflow() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    // Setup
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    let state_store = StateStore::open(&config)?;
    
    // Create test notes
    fs::create_dir_all(&notes_dir)?;
    fs::write(notes_dir.join("note1.md"), "# Note 1\n\nContent one.")?;
    fs::write(notes_dir.join("note2.md"), "# Note 2\n\nContent two.")?;
    
    // Discover files
    let files = discover_files(&notes_dir)?;
    assert_eq!(files.len(), 2);
    
    // Process each file
    for file in &files {
        // Parse
        let doc = parse_markdown_file(&file.path)?;
        assert!(!doc.chunks.is_empty());
        
        // Update state
        let modified_time = get_file_modified_time(&file.path)?;
        let hash = calculate_file_hash(&file.path)?;
        
        let relative_path = file.relative_path.to_str().unwrap();
        state_store.update_file_state(relative_path, modified_time, hash)?;
        
        // Verify state was updated
        let state = state_store.get_file_state(relative_path)?;
        assert!(state.is_some());
    }
    
    Ok(())
}

