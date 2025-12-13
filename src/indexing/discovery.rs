use crate::core::error::{Error, Result};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Represents a discovered file with its metadata
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub is_markdown: bool,
}

/// Discover all note files in a directory, respecting .gitignore rules
pub fn discover_files(root: &Path) -> Result<Vec<DiscoveredFile>> {
    if !root.exists() {
        return Err(Error::Config(format!(
            "Directory does not exist: {}",
            root.display()
        )));
    }

    if !root.is_dir() {
        return Err(Error::Config(format!(
            "Path is not a directory: {}",
            root.display()
        )));
    }

    let mut files = Vec::new();

    // Use ignore crate to walk directory respecting .gitignore
    let walker = WalkBuilder::new(root)
        .hidden(false) // We want to process hidden files (like .notes)
        .git_ignore(true)
        .git_exclude(true)
        .build();

    for result in walker {
        match result {
            Ok(entry) => {
                let path = entry.path();
                
                // Skip directories
                if path.is_dir() {
                    continue;
                }

                // Check if it's a supported notes file
                let is_markdown = is_notes_file(path);
                
                if is_markdown {
                    let relative_path = path
                        .strip_prefix(root)
                        .map_err(|e| Error::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to get relative path: {}", e),
                        )))?
                        .to_path_buf();

                    files.push(DiscoveredFile {
                        path: path.to_path_buf(),
                        relative_path,
                        is_markdown: true,
                    });
                }
            }
            Err(err) => {
                // Log but continue - some files might be inaccessible
                eprintln!("Warning: Failed to access file: {}", err);
            }
        }
    }

    Ok(files)
}

/// Check if a file is a supported notes file based on extension
pub fn is_notes_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_lowercase().as_str(),
                "md" | "markdown" | "mdown" | "mkd" | "mkdn" | "txt"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_is_notes_file() {
        assert!(is_notes_file(Path::new("test.md")));
        assert!(is_notes_file(Path::new("test.MD")));
        assert!(is_notes_file(Path::new("test.markdown")));
        assert!(is_notes_file(Path::new("test.mdown")));
        assert!(is_notes_file(Path::new("test.mkd")));
        assert!(is_notes_file(Path::new("test.mkdn")));
        assert!(is_notes_file(Path::new("test.txt")));
        assert!(!is_notes_file(Path::new("test")));
        assert!(!is_notes_file(Path::new("test.js")));
    }

    #[test]
    fn test_discover_files_basic() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        fs::write(test_dir.join("file1.md"), "# Test").unwrap();
        fs::write(test_dir.join("file2.md"), "# Test").unwrap();
        fs::write(test_dir.join("file3.txt"), "Not markdown").unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 3);
        
        let file_names: Vec<String> = files.iter()
            .map(|f| f.relative_path.to_str().unwrap().to_string())
            .collect();
        assert!(file_names.contains(&"file1.md".to_string()));
        assert!(file_names.contains(&"file2.md".to_string()));
        assert!(file_names.contains(&"file3.txt".to_string()));
    }

    #[test]
    fn test_discover_files_subdirectories() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        let subdir = test_dir.join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        fs::write(test_dir.join("file1.md"), "# Test").unwrap();
        fs::write(subdir.join("file2.md"), "# Test").unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_discover_files_various_extensions() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        fs::write(test_dir.join("file1.md"), "# Test").unwrap();
        fs::write(test_dir.join("file2.markdown"), "# Test").unwrap();
        fs::write(test_dir.join("file3.mdown"), "# Test").unwrap();
        fs::write(test_dir.join("file4.mkd"), "# Test").unwrap();
        fs::write(test_dir.join("file5.mkdn"), "# Test").unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn test_discover_files_case_insensitive() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        fs::write(test_dir.join("file1.MD"), "# Test").unwrap();
        fs::write(test_dir.join("file2.Markdown"), "# Test").unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_discover_files_relative_paths() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        let subdir = test_dir.join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        fs::write(test_dir.join("root.md"), "# Test").unwrap();
        fs::write(subdir.join("nested.md"), "# Test").unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 2);

        // Check relative paths
        let relative_paths: Vec<String> = files.iter()
            .map(|f| f.relative_path.to_str().unwrap().to_string())
            .collect();
        assert!(relative_paths.contains(&"root.md".to_string()));
        assert!(relative_paths.contains(&"subdir/nested.md".to_string()) || 
                relative_paths.contains(&"subdir\\nested.md".to_string())); // Windows vs Unix
    }

    #[test]
    fn test_discover_files_nonexistent_directory() {
        let result = discover_files(Path::new("/nonexistent/directory"));
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_files_file_instead_of_directory() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("file.txt");
        fs::write(&test_file, "content").unwrap();

        let result = discover_files(&test_file);
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_files_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("empty");
        fs::create_dir_all(&test_dir).unwrap();

        let files = discover_files(&test_dir).unwrap();
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_discover_files_all_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("notes");
        fs::create_dir_all(&test_dir).unwrap();

        let files = discover_files(&test_dir).unwrap();
        for file in &files {
            assert!(file.is_markdown);
        }
    }
}

