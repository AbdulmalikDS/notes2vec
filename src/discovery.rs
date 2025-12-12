use crate::error::{Error, Result};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Represents a discovered file with its metadata
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub is_markdown: bool,
}

/// Discover all Markdown files in a directory, respecting .gitignore rules
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

                // Check if it's a Markdown file
                let is_markdown = is_markdown_file(path);
                
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

/// Check if a file is a Markdown file based on extension
fn is_markdown_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_lowercase().as_str(),
                "md" | "markdown" | "mdown" | "mkd" | "mkdn"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_is_markdown_file() {
        assert!(is_markdown_file(Path::new("test.md")));
        assert!(is_markdown_file(Path::new("test.MD")));
        assert!(is_markdown_file(Path::new("test.markdown")));
        assert!(!is_markdown_file(Path::new("test.txt")));
        assert!(!is_markdown_file(Path::new("test")));
    }
}

