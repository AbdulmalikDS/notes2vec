use crate::core::config::Config;
use crate::core::error::{Error, Result};
use redb::{Database, ReadableTable, TableDefinition};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::SystemTime;

/// Table definition for file state tracking
/// Using &str for both key and value (JSON serialized)
const FILE_STATE_TABLE: TableDefinition<&str, &str> = TableDefinition::new("file_state");

// Stored in FILE_STATE_TABLE as a JSON string; used to detect model changes and force re-index.
const META_MODEL_ID_KEY: &str = "__notes2vec_meta_model_id__";

/// State information for a file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileState {
    /// Last modification time
    pub last_modified: u64,
    /// SHA256 hash of file contents
    pub content_hash: String,
    /// Timestamp when file was last indexed
    pub indexed_at: u64,
}

impl FileState {
    /// Create a new file state
    pub fn new(last_modified: u64, content_hash: String) -> Self {
        Self {
            last_modified,
            content_hash,
            indexed_at: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Serialize to JSON string
    fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| Error::Database(format!("Failed to serialize file state: {}", e)))
    }

    /// Deserialize from JSON string
    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::Database(format!("Failed to deserialize file state: {}", e)))
    }
}

/// State store for tracking file changes
pub struct StateStore {
    db: Database,
}

impl StateStore {
    /// Open or create the state store
    pub fn open(config: &Config) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.state_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create or open the database
        // Database::create will create a new database or open existing one
        let db = if config.state_path.exists() {
            Database::open(&config.state_path)
                .map_err(|e| {
                    let msg = e.to_string();
                    if msg.to_lowercase().contains("lock") {
                        Error::Database("State database is locked. Another notes2vec process may be running. Close other instances and try again.".to_string())
                    } else {
                        Error::Database(format!("Failed to open state database: {}", e))
                    }
                })?
        } else {
            Database::create(&config.state_path)
                .map_err(|e| Error::Database(format!("Failed to create state database: {}", e)))?
        };

        // Initialize table (this is safe even if table already exists)
        let write_txn = db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;
        {
            let _table = write_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;
        }
        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(Self { db })
    }

    /// Get the state of a file
    pub fn get_file_state(&self, file_path: &str) -> Result<Option<FileState>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let result = match table.get(file_path).map_err(|e| {
            Error::Database(format!("Failed to get file state: {}", e))
        })? {
            Some(guard) => {
                // Extract the value string before dropping the guard
                let json_str = guard.value().to_string();
                FileState::from_json(&json_str).map(Some)
            }
            None => Ok(None),
        };
        
        result
    }

    /// Update the state of a file
    pub fn update_file_state(
        &self,
        file_path: &str,
        last_modified: u64,
        content_hash: String,
    ) -> Result<()> {
        let write_txn = self.db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;

            let state = FileState::new(last_modified, content_hash);
            let json_str = state.to_json()?;
            table.insert(file_path, json_str.as_str()).map_err(|e| {
                Error::Database(format!("Failed to insert file state: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    }

    /// Remove a file from the state store
    pub fn remove_file(&self, file_path: &str) -> Result<()> {
        let write_txn = self.db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;

            table.remove(file_path).map_err(|e| {
                Error::Database(format!("Failed to remove file state: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    }

    /// Check if a file has changed since last indexing
    pub fn has_file_changed(
        &self,
        file_path: &str,
        current_modified: u64,
        current_hash: &str,
    ) -> Result<bool> {
        match self.get_file_state(file_path)? {
            Some(state) => {
                // File has changed if modification time or hash differs
                Ok(state.last_modified != current_modified || state.content_hash != current_hash)
            }
            None => {
                // File not in state store, consider it changed (needs indexing)
                Ok(true)
            }
        }
    }

    pub fn get_model_id(&self) -> Result<Option<String>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let v = table.get(META_MODEL_ID_KEY).map_err(|e| {
            Error::Database(format!("Failed to get model id: {}", e))
        })?;

        match v {
            Some(guard) => Ok(Some(guard.value().to_string())),
            None => Ok(None),
        }
    }

    pub fn set_model_id(&self, model_id: &str) -> Result<()> {
        let write_txn = self.db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(FILE_STATE_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;
            table.insert(META_MODEL_ID_KEY, model_id).map_err(|e| {
                Error::Database(format!("Failed to store model id: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    }
}

/// Calculate SHA256 hash of file contents
pub fn calculate_file_hash(path: &Path) -> Result<String> {
    use std::io::Read;
    
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    
    Ok(format!("{:x}", hasher.finalize()))
}

/// Get file modification time as Unix timestamp
pub fn get_file_modified_time(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?;
    let duration = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to get modification time: {}", e),
        )))?;
    Ok(duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_file_state_creation() {
        let state = FileState::new(12345, "abc123".to_string());
        assert_eq!(state.last_modified, 12345);
        assert_eq!(state.content_hash, "abc123");
        assert!(state.indexed_at > 0);
    }

    #[test]
    fn test_file_state_serialization() {
        let state = FileState::new(12345, "abc123".to_string());
        
        let json = state.to_json().unwrap();
        assert!(json.contains("12345"));
        assert!(json.contains("abc123"));

        let deserialized = FileState::from_json(&json).unwrap();
        assert_eq!(deserialized.last_modified, state.last_modified);
        assert_eq!(deserialized.content_hash, state.content_hash);
    }

    #[test]
    fn test_state_store_open_and_init() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let _store = StateStore::open(&config).unwrap();
        // Should not panic
        assert!(true);
    }

    #[test]
    fn test_state_store_update_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = StateStore::open(&config).unwrap();

        // File should not exist initially
        let state = store.get_file_state("test.md").unwrap();
        assert!(state.is_none());

        // Update state
        store.update_file_state("test.md", 12345, "hash123".to_string()).unwrap();

        // File should exist now
        let state = store.get_file_state("test.md").unwrap();
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.last_modified, 12345);
        assert_eq!(state.content_hash, "hash123");
    }

    #[test]
    fn test_state_store_has_file_changed() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = StateStore::open(&config).unwrap();

        // New file should be considered changed
        assert!(store.has_file_changed("new.md", 12345, "hash1").unwrap());

        // Update state
        store.update_file_state("new.md", 12345, "hash1".to_string()).unwrap();

        // Same file should not be changed
        assert!(!store.has_file_changed("new.md", 12345, "hash1").unwrap());

        // Different modification time should be changed
        assert!(store.has_file_changed("new.md", 12346, "hash1").unwrap());

        // Different hash should be changed
        assert!(store.has_file_changed("new.md", 12345, "hash2").unwrap());
    }

    #[test]
    fn test_state_store_remove_file() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = StateStore::open(&config).unwrap();

        // Add file
        store.update_file_state("test.md", 12345, "hash123".to_string()).unwrap();
        assert!(store.get_file_state("test.md").unwrap().is_some());

        // Remove file
        store.remove_file("test.md").unwrap();
        assert!(store.get_file_state("test.md").unwrap().is_none());
    }

    #[test]
    fn test_state_store_remove_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = StateStore::open(&config).unwrap();

        // Removing non-existent file should not error
        store.remove_file("nonexistent.md").unwrap();
    }

    #[test]
    fn test_calculate_file_hash() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        // Create file with content
        fs::write(&test_file, "Hello, world!").unwrap();

        let hash1 = calculate_file_hash(&test_file).unwrap();
        assert!(!hash1.is_empty());
        assert_eq!(hash1.len(), 64); // SHA256 produces 64 hex characters

        // Same content should produce same hash
        let hash2 = calculate_file_hash(&test_file).unwrap();
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        fs::write(&test_file, "Different content").unwrap();
        let hash3 = calculate_file_hash(&test_file).unwrap();
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_calculate_file_hash_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("empty.txt");

        fs::write(&test_file, "").unwrap();

        let hash = calculate_file_hash(&test_file).unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_calculate_file_hash_large_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("large.txt");

        // Create a file larger than buffer size (8192 bytes)
        let content = "x".repeat(10000);
        fs::write(&test_file, content).unwrap();

        let hash = calculate_file_hash(&test_file).unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_get_file_modified_time() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        fs::write(&test_file, "Test content").unwrap();

        let time1 = get_file_modified_time(&test_file).unwrap();
        assert!(time1 > 0);

        // Wait a bit and modify file
        std::thread::sleep(std::time::Duration::from_millis(100));
        fs::write(&test_file, "Modified content").unwrap();

        let time2 = get_file_modified_time(&test_file).unwrap();
        assert!(time2 >= time1);
    }

    #[test]
    fn test_get_file_modified_time_nonexistent() {
        let result = get_file_modified_time(std::path::Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}

