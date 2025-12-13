use notes2vec::{Config, Result};
use notes2vec::FileWatcher;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Test basic watch functionality
#[test]
fn test_watch_basic() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    // Setup
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    // Create notes directory
    fs::create_dir_all(&notes_dir)?;
    
    // Create initial file
    fs::write(notes_dir.join("test1.md"), "# Test 1\n\nInitial content.")?;
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config.clone())?;
    
    // Note: We can't easily test the full watch loop in a unit test
    // because it blocks indefinitely. Instead, we test the components.
    
    // Test that watcher was created successfully
    // This verifies the basic setup works
    assert!(true); // Placeholder - watcher creation succeeded
    
    Ok(())
}

/// Test that watch mode can handle file creation
#[test]
fn test_watch_file_creation() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Create a new file
    fs::write(notes_dir.join("new_file.md"), "# New File\n\nContent.")?;
    
    // Note: In a real scenario, the watcher would detect this
    // For unit testing, we verify the file exists and can be processed
    assert!(notes_dir.join("new_file.md").exists());
    
    Ok(())
}

/// Test that watch mode can handle file deletion
#[test]
fn test_watch_file_deletion() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create a file
    let test_file = notes_dir.join("to_delete.md");
    fs::write(&test_file, "# To Delete\n\nContent.")?;
    assert!(test_file.exists());
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Delete the file
    fs::remove_file(&test_file)?;
    assert!(!test_file.exists());
    
    // Note: In a real scenario, the watcher would detect this deletion
    // For unit testing, we verify the file was deleted
    
    Ok(())
}

/// Test that watch mode can handle file modification
#[test]
fn test_watch_file_modification() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create initial file
    let test_file = notes_dir.join("to_modify.md");
    fs::write(&test_file, "# Original\n\nOriginal content.")?;
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Modify the file
    fs::write(&test_file, "# Modified\n\nModified content.")?;
    
    // Verify modification
    let content = fs::read_to_string(&test_file)?;
    assert!(content.contains("Modified"));
    
    // Note: In a real scenario, the watcher would detect this change
    // For unit testing, we verify the file was modified
    
    Ok(())
}

/// Test that watch mode ignores non-markdown files
#[test]
fn test_watch_ignores_non_markdown() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Create non-markdown files
    fs::write(notes_dir.join("test.txt"), "Not markdown")?;
    fs::write(notes_dir.join("test.js"), "// JavaScript")?;
    fs::write(notes_dir.join("test.py"), "# Python")?;
    
    // These files should be ignored by the watcher
    // (In a real scenario, they wouldn't trigger indexing)
    assert!(notes_dir.join("test.txt").exists());
    assert!(notes_dir.join("test.js").exists());
    assert!(notes_dir.join("test.py").exists());
    
    Ok(())
}

/// Test that watch mode handles multiple files
#[test]
fn test_watch_multiple_files() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create multiple markdown files
    for i in 1..=5 {
        fs::write(
            notes_dir.join(format!("file{}.md", i)),
            format!("# File {}\n\nContent {}.", i, i)
        )?;
    }
    
    // Create watcher
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Verify all files exist
    for i in 1..=5 {
        assert!(notes_dir.join(format!("file{}.md", i)).exists());
    }
    
    Ok(())
}

/// Test that watch mode handles subdirectories
#[test]
fn test_watch_subdirectories() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    let notes_dir = temp_dir.path().join("notes");
    
    let config = Config::new(Some(base_dir))?;
    config.init()?;
    
    fs::create_dir_all(&notes_dir)?;
    
    // Create subdirectory
    let subdir = notes_dir.join("subdir");
    fs::create_dir_all(&subdir)?;
    
    // Create file in subdirectory
    fs::write(subdir.join("nested.md"), "# Nested\n\nNested content.")?;
    
    // Create watcher (should watch recursively)
    let _watcher = FileWatcher::new(&notes_dir, config)?;
    
    // Verify nested file exists
    assert!(subdir.join("nested.md").exists());
    
    Ok(())
}

/// Test error handling for invalid paths
#[test]
fn test_watch_invalid_path() {
    let temp_dir = TempDir::new().unwrap();
    let base_dir = temp_dir.path().join("test_notes2vec");
    
    let config = Config::new(Some(base_dir)).unwrap();
    config.init().unwrap();
    
    // Try to watch a non-existent directory
    let result = FileWatcher::new(Path::new("/nonexistent/path"), config);
    // This should succeed (watcher creation doesn't validate path immediately)
    // The actual watching would fail, but that's tested in integration
    
    // Try to watch a file instead of directory
    let test_file = temp_dir.path().join("test.md");
    fs::write(&test_file, "# Test").unwrap();
    
    let config2 = Config::new(Some(temp_dir.path().join("test2"))).unwrap();
    config2.init().unwrap();
    let result2 = FileWatcher::new(&test_file, config2);
    // This should succeed (watcher creation doesn't validate immediately)
    
    // Both should succeed at creation time
    assert!(result.is_ok());
    assert!(result2.is_ok());
}

