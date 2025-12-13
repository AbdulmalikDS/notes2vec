use crate::core::config::Config;
use crate::core::error::{Error, Result};
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
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| Error::Database(format!("Failed to serialize vector entry: {}", e)))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
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
                .map_err(|e| {
                    let msg = e.to_string();
                    if msg.to_lowercase().contains("lock") {
                        Error::Database("Vector database is locked. Another notes2vec process may be running. Close other instances and try again.".to_string())
                    } else {
                        Error::Database(format!("Failed to open vector database: {}", e))
                    }
                })?
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
    /// Optimized: Uses chunk_id prefix matching to avoid deserializing all entries
    pub fn remove_file(&self, file_path: &str) -> Result<usize> {
        // First, collect all chunk IDs to remove in a read transaction
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;
        
        let read_table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        // Collect all chunk IDs to remove
        // Since chunk_id format is "file_path:chunk_index", we can optimize by checking prefix
        let prefix = format!("{}:", file_path);
        let mut to_remove = Vec::new();
        
        for item in read_table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (key, _value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            let key_str = key.value();
            
            // Check if key starts with our file_path prefix (optimization: avoid deserialization)
            if key_str.starts_with(&prefix) {
                to_remove.push(key_str.to_string());
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
    /// Uses a min-heap to efficiently maintain top K results without storing all vectors
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(VectorEntry, f32)>> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        // Min-heap (via reversed ordering): smallest similarity at the top
        let mut heap: BinaryHeap<SimilarityEntry> = BinaryHeap::with_capacity(limit + 1);

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
                
                // Add to heap
                heap.push(SimilarityEntry(entry, similarity));
                
                // Keep only top K results
                if heap.len() > limit {
                    heap.pop(); // Remove smallest similarity
                }
            }
        }

        // Convert heap to sorted vector (descending similarity)
        let mut results: Vec<(VectorEntry, f32)> = heap
            .into_iter()
            .map(|se| (se.0, se.1))
            .collect();
        
        // Sort descending by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Search, but only consider vectors belonging to the provided set of file paths.
    /// This is significantly faster for the TUI, and prevents cross-folder results.
    pub fn search_scoped(
        &self,
        query_embedding: &[f32],
        limit: usize,
        allowed_files: &std::collections::HashSet<String>,
    ) -> Result<Vec<(VectorEntry, f32)>> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let mut heap: BinaryHeap<SimilarityEntry> = BinaryHeap::with_capacity(limit + 1);

        for item in table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (key, value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;

            // Key format: "file_path:chunk_index" — check scope before deserializing.
            let key_str = key.value();
            let file_part = key_str.split(':').next().unwrap_or("");
            if !allowed_files.contains(file_part) {
                continue;
            }

            let json_str = value.value().to_string();
            if let Ok(entry) = VectorEntry::from_json(&json_str) {
                let similarity = cosine_similarity(query_embedding, &entry.embedding);
                heap.push(SimilarityEntry(entry, similarity));
                if heap.len() > limit {
                    heap.pop();
                }
            }
        }

        let mut results: Vec<(VectorEntry, f32)> = heap.into_iter().map(|se| (se.0, se.1)).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    /// Get all vectors for a specific file
    /// Optimized: Uses chunk_id prefix matching to avoid deserializing non-matching entries
    pub fn get_file_vectors(&self, file_path: &str) -> Result<Vec<VectorEntry>> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let mut results = Vec::new();
        let prefix = format!("{}:", file_path);

        for item in table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (key, value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            
            // Optimize: Check prefix before deserializing
            if key.value().starts_with(&prefix) {
                let json_str = value.value().to_string();
                if let Ok(entry) = VectorEntry::from_json(&json_str) {
                    results.push(entry);
                }
            }
        }

        // Sort by chunk_index
        results.sort_by_key(|e| e.chunk_index);

        Ok(results)
    }

    /// Get count of unique indexed files
    pub fn get_file_count(&self) -> Result<usize> {
        let read_txn = self.db.begin_read().map_err(|e| {
            Error::Database(format!("Failed to begin read transaction: {}", e))
        })?;

        let table = read_txn.open_table(VECTORS_TABLE).map_err(|e| {
            Error::Database(format!("Failed to open table: {}", e))
        })?;

        let mut unique_files = std::collections::HashSet::new();

        for item in table.iter().map_err(|e| {
            Error::Database(format!("Failed to iterate table: {}", e))
        })? {
            let (key, _value) = item.map_err(|e| {
                Error::Database(format!("Failed to read table item: {}", e))
            })?;
            
            // Extract file path from chunk_id (format: "file_path:chunk_index")
            if let Some(file_path) = key.value().split(':').next() {
                unique_files.insert(file_path.to_string());
            }
        }

        Ok(unique_files.len())
    }
}

/// Helper struct for maintaining top-K search results using a min-heap
struct SimilarityEntry(VectorEntry, f32);

impl PartialEq for SimilarityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Eq for SimilarityEntry {}

impl PartialOrd for SimilarityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse ordering so BinaryHeap (a max-heap) behaves like a min-heap by similarity.
        other.1.partial_cmp(&self.1)
    }
}

impl std::cmp::Ord for SimilarityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
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
    use crate::core::config::Config;
    use tempfile::TempDir;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Zero vectors
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Different lengths (should return 0.0)
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Negative values
        let a = vec![1.0, -1.0];
        let b = vec![-1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim < 0.0); // Should be negative (opposite directions)
    }

    #[test]
    fn test_vector_entry_creation() {
        let entry = VectorEntry::new(
            "test.md".to_string(),
            0,
            vec![1.0, 2.0, 3.0],
            "Test text".to_string(),
            "Context".to_string(),
            1,
            10,
        );

        assert_eq!(entry.file_path, "test.md");
        assert_eq!(entry.chunk_index, 0);
        assert_eq!(entry.embedding.len(), 3);
        assert_eq!(entry.text, "Test text");
        assert_eq!(entry.context, "Context");
        assert_eq!(entry.start_line, 1);
        assert_eq!(entry.end_line, 10);
    }

    #[test]
    fn test_vector_entry_chunk_id() {
        let entry1 = VectorEntry::new(
            "file1.md".to_string(),
            0,
            vec![1.0],
            "text".to_string(),
            "context".to_string(),
            1,
            1,
        );
        assert_eq!(entry1.chunk_id(), "file1.md:0");

        let entry2 = VectorEntry::new(
            "file2.md".to_string(),
            42,
            vec![1.0],
            "text".to_string(),
            "context".to_string(),
            1,
            1,
        );
        assert_eq!(entry2.chunk_id(), "file2.md:42");
    }

    #[test]
    fn test_vector_entry_serialization() {
        let entry = VectorEntry::new(
            "test.md".to_string(),
            5,
            vec![0.1, 0.2, 0.3],
            "Test content".to_string(),
            "Section > Subsection".to_string(),
            10,
            20,
        );

        // Test serialization
        let json = entry.to_json().unwrap();
        assert!(json.contains("test.md"));
        assert!(json.contains("Test content"));

        // Test deserialization
        let deserialized = VectorEntry::from_json(&json).unwrap();
        assert_eq!(deserialized.file_path, entry.file_path);
        assert_eq!(deserialized.chunk_index, entry.chunk_index);
        assert_eq!(deserialized.embedding, entry.embedding);
        assert_eq!(deserialized.text, entry.text);
        assert_eq!(deserialized.context, entry.context);
        assert_eq!(deserialized.start_line, entry.start_line);
        assert_eq!(deserialized.end_line, entry.end_line);
    }

    #[test]
    fn test_vector_store_insert_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        let entry = VectorEntry::new(
            "test.md".to_string(),
            0,
            vec![0.1, 0.2, 0.3, 0.4],
            "Test text".to_string(),
            "Context".to_string(),
            1,
            10,
        );

        // Insert
        store.insert(&entry).unwrap();

        // Get
        let retrieved = store.get("test.md:0").unwrap();
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.file_path, "test.md");
        assert_eq!(retrieved_entry.chunk_index, 0);
        assert_eq!(retrieved_entry.text, "Test text");
    }

    #[test]
    fn test_vector_store_get_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        let result = store.get("nonexistent.md:0").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_vector_store_remove_file() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        // Insert multiple entries for same file
        for i in 0..5 {
            let entry = VectorEntry::new(
                "test.md".to_string(),
                i,
                vec![0.1, 0.2, 0.3],
                format!("Chunk {}", i),
                "Context".to_string(),
                1,
                10,
            );
            store.insert(&entry).unwrap();
        }

        // Insert entry for different file
        let other_entry = VectorEntry::new(
            "other.md".to_string(),
            0,
            vec![0.1, 0.2, 0.3],
            "Other text".to_string(),
            "Context".to_string(),
            1,
            10,
        );
        store.insert(&other_entry).unwrap();

        // Verify entries exist
        assert!(store.get("test.md:0").unwrap().is_some());
        assert!(store.get("test.md:4").unwrap().is_some());
        assert!(store.get("other.md:0").unwrap().is_some());

        // Remove file
        let removed_count = store.remove_file("test.md").unwrap();
        assert_eq!(removed_count, 5);

        // Verify test.md entries are gone
        assert!(store.get("test.md:0").unwrap().is_none());
        assert!(store.get("test.md:4").unwrap().is_none());

        // Verify other.md entry still exists
        assert!(store.get("other.md:0").unwrap().is_some());
    }

    #[test]
    fn test_vector_store_remove_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        // Remove non-existent file should return 0
        let removed_count = store.remove_file("nonexistent.md").unwrap();
        assert_eq!(removed_count, 0);
    }

    #[test]
    fn test_vector_store_get_file_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        // Insert multiple chunks for same file
        let chunk_indices = vec![0, 2, 5, 10];
        for &idx in &chunk_indices {
            let entry = VectorEntry::new(
                "test.md".to_string(),
                idx,
                vec![0.1, 0.2, 0.3],
                format!("Chunk {}", idx),
                "Context".to_string(),
                1,
                10,
            );
            store.insert(&entry).unwrap();
        }

        // Get all vectors for file
        let vectors = store.get_file_vectors("test.md").unwrap();
        assert_eq!(vectors.len(), 4);

        // Verify they're sorted by chunk_index
        for i in 0..vectors.len() - 1 {
            assert!(vectors[i].chunk_index <= vectors[i + 1].chunk_index);
        }

        // Verify chunk indices match
        let retrieved_indices: Vec<usize> = vectors.iter().map(|v| v.chunk_index).collect();
        assert_eq!(retrieved_indices, chunk_indices);
    }

    #[test]
    fn test_vector_store_get_file_count() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        // Initially empty
        assert_eq!(store.get_file_count().unwrap(), 0);

        // Add entries for multiple files
        let files = vec!["file1.md", "file2.md", "file3.md"];
        for (_file_idx, file) in files.iter().enumerate() {
            for chunk_idx in 0..3 {
                let entry = VectorEntry::new(
                    file.to_string(),
                    chunk_idx,
                    vec![0.1, 0.2, 0.3],
                    format!("Chunk {}", chunk_idx),
                    "Context".to_string(),
                    1,
                    10,
                );
                store.insert(&entry).unwrap();
            }
        }

        // Should count 3 unique files
        assert_eq!(store.get_file_count().unwrap(), 3);

        // Remove one file
        store.remove_file("file1.md").unwrap();
        assert_eq!(store.get_file_count().unwrap(), 2);
    }

    #[test]
    fn test_vector_store_search() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        // Create query vector
        let query = vec![1.0, 0.0, 0.0];

        // Insert vectors with different similarities
        // Vector similar to query (high similarity)
        let similar_entry = VectorEntry::new(
            "similar.md".to_string(),
            0,
            vec![1.0, 0.0, 0.0], // Same as query
            "Similar content".to_string(),
            "Context".to_string(),
            1,
            10,
        );
        store.insert(&similar_entry).unwrap();

        // Vector orthogonal to query (low similarity)
        let different_entry = VectorEntry::new(
            "different.md".to_string(),
            0,
            vec![0.0, 1.0, 0.0], // Orthogonal to query
            "Different content".to_string(),
            "Context".to_string(),
            1,
            10,
        );
        store.insert(&different_entry).unwrap();

        // Search with limit
        let results = store.search(&query, 5).unwrap();
        assert_eq!(results.len(), 2);

        // Results should be sorted by similarity (descending)
        assert!(results[0].1 >= results[1].1);

        // First result should be the similar one
        assert_eq!(results[0].0.file_path, "similar.md");
        assert!(results[0].1 > 0.9); // High similarity
    }

    #[test]
    fn test_vector_store_search_with_limit() {
        let temp_dir = TempDir::new().unwrap();
        let base_dir = temp_dir.path().join("test_notes2vec");
        let config = Config::new(Some(base_dir)).unwrap();
        config.init().unwrap();

        let store = VectorStore::open(&config).unwrap();

        let query = vec![1.0, 0.0, 0.0];

        // Insert many vectors
        for i in 0..20 {
            let entry = VectorEntry::new(
                format!("file{}.md", i),
                0,
                vec![0.1 * i as f32, 0.2, 0.3],
                format!("Content {}", i),
                "Context".to_string(),
                1,
                10,
            );
            store.insert(&entry).unwrap();
        }

        // Search with limit
        let results = store.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);

        // Verify sorted by similarity
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }
}

