use crate::config::Config;
use crate::error::{Error, Result};
use redb::{Database, ReadableTable, TableDefinition};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::SystemTime;

/// Table definition for file state tracking
/// Using &str for both key and value (JSON serialized)
const FILE_STATE_TABLE: TableDefinition<&str, &str> = TableDefinition::new("file_state");

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

        let db = Database::create(&config.state_path)
            .map_err(|e| Error::Database(format!("Failed to create state database: {}", e)))?;

        // Initialize table
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

