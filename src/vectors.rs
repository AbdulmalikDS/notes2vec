use crate::config::Config;
use crate::error::{Error, Result};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

/// Table definition for vector storage
/// Key: chunk_id (format: "file_path:chunk_index")
/// Value: JSON serialized VectorEntry
const VECTORS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("vectors");

/// Metadata for a vector entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// File path (relative)
    pub file_path: String,
    /// Chunk index within the file
    pub chunk_index: usize,
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Text content of the chunk
    pub text: String,
    /// Context (e.g., "Document > Section")
    pub context: String,
    /// Start line in source file
    pub start_line: usize,
    /// End line in source file
    pub end_line: usize,
}

impl VectorEntry {
    /// Create a new vector entry
    pub fn new(
        file_path: String,
        chunk_index: usize,
        embedding: Vec<f32>,
        text: String,
        context: String,
        start_line: usize,
        end_line: usize,
    ) -> Self {
        Self {
            file_path,
            chunk_index,
            embedding,
            text,
            context,
            start_line,
            end_line,
        }
    }

    /// Get a unique ID for this chunk
    pub fn chunk_id(&self) -> String {
        format!("{}:{}", self.file_path, self.chunk_index)
    }

    /// Serialize to JSON
    fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| Error::Database(format!("Failed to serialize vector entry: {}", e)))
    }

    /// Deserialize from JSON
    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::Database(format!("Failed to deserialize vector entry: {}", e)))
    }
}

/// Vector store for managing embeddings
pub struct VectorStore {
    db: Database,
}

impl VectorStore {
    /// Open or create the vector store
    pub fn open(config: &Config) -> Result<Self> {
        // Use the database directory for vector storage
        let db_path = config.database_dir.join("vectors.redb");
        
        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create or open the database
        let db = if db_path.exists() {
            Database::open(&db_path)
                .map_err(|e| Error::Database(format!("Failed to open vector database: {}", e)))?
        } else {
            Database::create(&db_path)
                .map_err(|e| Error::Database(format!("Failed to create vector database: {}", e)))?
        };

        // Initialize table
        let write_txn = db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;
        {
            let _table = write_txn.open_table(VECTORS_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;
        }
        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(Self { db })
    }

    /// Insert or update a vector entry
    pub fn insert(&self, entry: &VectorEntry) -> Result<()> {
        let write_txn = self.db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(VECTORS_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;

            let chunk_id = entry.chunk_id();
            let json_str = entry.to_json()?;
            table.insert(chunk_id.as_str(), json_str.as_str()).map_err(|e| {
                Error::Database(format!("Failed to insert vector entry: {}", e))
            })?;
        }

        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(())
    }

    /// Get a vector entry by chunk ID
    pub fn get(&self, chunk_id: &str) -> Result<Option<VectorEntry>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let guard_option = table.get(chunk_id).map_err(|e| {
            Error::Database(format!("Failed to get vector entry: {}", e))
        })?;

        let result = match guard_option {
            Some(guard) => {
                // Extract the value string before dropping the guard
                let json_str = guard.value().to_string();
                drop(guard); // Explicitly drop guard
                VectorEntry::from_json(&json_str).map(Some)
            }
            None => Ok(None),
        };
        
        result
    }

    /// Remove all vectors for a specific file
    pub fn remove_file(&self, file_path: &str) -> Result<usize> {
        // First, collect all chunk IDs to remove in a read transaction
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;
        
        let read_table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        // Collect all chunk IDs to remove
        let mut to_remove = Vec::new();
        for item in read_table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (key, value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            let json_str = value.value().to_string();
            if let Ok(entry) = VectorEntry::from_json(&json_str) {
                if entry.file_path == file_path {
                    to_remove.push(key.value().to_string());
                }
            }
        }
        
        // Drop read transaction before starting write transaction
        drop(read_table);
        drop(read_txn);

        // Now remove entries in a write transaction
        if to_remove.is_empty() {
            return Ok(0);
        }

        let write_txn = self.db.begin_write().map_err(|e| {
            Error::Database(format!("Failed to begin write transaction: {}", e))
        })?;

        {
            let mut table = write_txn.open_table(VECTORS_TABLE).map_err(|e| {
                Error::Database(format!("Failed to open table: {}", e))
            })?;

            // Remove entries
            for chunk_id in &to_remove {
                table.remove(chunk_id.as_str()).map_err(|e| {
                    Error::Database(format!("Failed to remove vector entry: {}", e))
                })?;
            }
        }

        write_txn.commit().map_err(|e| {
            Error::Database(format!("Failed to commit transaction: {}", e))
        })?;

        Ok(to_remove.len())
    }

    /// Search for similar vectors using cosine similarity
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(VectorEntry, f32)>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let mut results = Vec::new();

        // Iterate through all vectors and compute similarity
        for item in table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (_key, value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            let json_str = value.value().to_string();
            if let Ok(entry) = VectorEntry::from_json(&json_str) {
                let similarity = cosine_similarity(query_embedding, &entry.embedding);
                results.push((entry, similarity));
            }
        }

        // Sort by similarity (descending) and take top results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Get all vectors for a specific file
    pub fn get_file_vectors(&self, file_path: &str) -> Result<Vec<VectorEntry>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let mut results = Vec::new();

        for item in table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (_key, value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            let json_str = value.value().to_string();
            if let Ok(entry) = VectorEntry::from_json(&json_str) {
                if entry.file_path == file_path {
                    results.push(entry);
                }
            }
        }

        // Sort by chunk_index
        results.sort_by_key(|e| e.chunk_index);

        Ok(results)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }
}

