use crate::core::error::{Error, Result};
use crate::search::model::EmbeddingModel;
use crate::storage::vectors::{VectorEntry, VectorStore};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// Search configuration constants
const SEARCH_CANDIDATES_LIMIT: usize = 200;      // Number of candidates to fetch for unconstrained search
const SCOPED_SEARCH_CANDIDATES_LIMIT: usize = 500; // Number of candidates for scoped search
pub const MAX_RESULTS_DISPLAYED: usize = 20;     // Maximum number of results to display (top 20 passages)
const MAX_RESULTS_PER_FILE: usize = 5;           // Maximum results per file (allows multiple chunks from same file)

// Lexical boost values for search results
const LEXICAL_BOOST_PATH: f32 = 0.05;   // Boost for filename matches
const LEXICAL_BOOST_CONTEXT: f32 = 0.10; // Boost for context matches
const LEXICAL_BOOST_TEXT: f32 = 0.15;    // Boost for text content matches

/// Perform semantic search with lexical boosting and deduplication
pub fn perform_search(
    query: &str,
    model: &EmbeddingModel,
    vector_store: &VectorStore,
    active_files: &HashSet<String>,
) -> Result<Vec<(VectorEntry, f32)>> {
    let (file_filter, semantic_query) = parse_file_filter_query(query);

    if semantic_query.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Keep a lowercase copy for small lexical boosting before we move the String.
    let q_lower = semantic_query.to_lowercase();
    let query_texts = vec![semantic_query];
    let query_embeddings = model.embed_queries(&query_texts)?;

    if query_embeddings.is_empty() {
        return Err(Error::Model("Failed to generate query embedding".to_string()));
    }

    let query_embedding = &query_embeddings[0];
    // Get more candidates, then scope + boost + dedupe to top results (better UX).
    // For scoped searches, fetch even more candidates to ensure we get enough results
    let mut results = if active_files.is_empty() {
        vector_store.search(query_embedding, SEARCH_CANDIDATES_LIMIT)?
    } else {
        // For scoped search, fetch enough candidates to get top passages
        // Multiply by MAX_RESULTS_PER_FILE to ensure we get multiple chunks per file
        let candidate_limit = (MAX_RESULTS_DISPLAYED * MAX_RESULTS_PER_FILE).max(SCOPED_SEARCH_CANDIDATES_LIMIT);
        vector_store.search_scoped(query_embedding, candidate_limit, active_files)?
    };

    // Optional: limit results to a specific file (or partial filename).
    if let Some(filter) = file_filter {
        results.retain(|(entry, _)| path_matches_filter(&entry.file_path, &filter));
    }

    // Small lexical boost for obvious matches (helps short queries like "Agenda")
    // Optimized: Use case-insensitive matching helper to reduce allocations
    if !q_lower.is_empty() {
        for (entry, sim) in results.iter_mut() {
            let mut bonus = 0.0f32;
            // Use efficient case-insensitive contains (only allocates when needed)
            if contains_case_insensitive(&entry.file_path, &q_lower) {
                bonus += LEXICAL_BOOST_PATH;
            }
            if contains_case_insensitive(&entry.context, &q_lower) {
                bonus += LEXICAL_BOOST_CONTEXT;
            }
            if contains_case_insensitive(&entry.text, &q_lower) {
                bonus += LEXICAL_BOOST_TEXT;
            }
            *sim = (*sim + bonus).min(1.0);
        }
    }

    // Smart deduplication: allow multiple results per file (up to MAX_RESULTS_PER_FILE)
    // This allows users to see multiple relevant chunks from the same file
    // Group results by file, keep top N per file, then take overall top results
    let mut results_by_file: HashMap<String, Vec<(VectorEntry, f32)>> = HashMap::new();
    
    for (entry, sim) in results {
        results_by_file
            .entry(entry.file_path.clone())
            .or_insert_with(Vec::new)
            .push((entry, sim));
    }
    
    // Sort each file's results by similarity (descending) and keep top N per file
    for file_results in results_by_file.values_mut() {
        file_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        file_results.truncate(MAX_RESULTS_PER_FILE);
    }
    
    // Flatten and sort all results by similarity
    let mut all_results: Vec<(VectorEntry, f32)> = results_by_file
        .into_values()
        .flatten()
        .collect();
    
    all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Return top 20 passages (or all if less than 20)
    all_results.truncate(MAX_RESULTS_DISPLAYED);

    Ok(all_results)
}

/// Parse query string to extract file filter and semantic query
pub fn parse_file_filter_query(raw: &str) -> (Option<String>, String) {
    let mut filter: Option<String> = None;
    let mut parts: Vec<&str> = Vec::new();

    for token in raw.split_whitespace() {
        if let Some(rest) = token.strip_prefix("file:") {
            if !rest.is_empty() {
                // Allow file:"name.md" and strip trailing punctuation like commas.
                let cleaned = rest
                    .trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';' || c == '.');
                filter = Some(cleaned.to_string());
                continue;
            }
        }
        parts.push(token);
    }

    (filter, parts.join(" "))
}

/// Case-insensitive contains check (optimized for ASCII, falls back to allocation for Unicode)
fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    // Fast path: if both strings are ASCII, use byte-level comparison without allocation
    if haystack.is_ascii() && needle.is_ascii() {
        let haystack_bytes = haystack.as_bytes();
        let needle_bytes = needle.as_bytes();
        haystack_bytes
            .windows(needle_bytes.len())
            .any(|window| {
                window.iter().zip(needle_bytes.iter()).all(|(&b, &n)| {
                    b.to_ascii_lowercase() == n.to_ascii_lowercase()
                })
            })
    } else {
        // Unicode path: must allocate for proper case-insensitive matching
        haystack.to_lowercase().contains(needle)
    }
}

/// Check if a file path matches a filter string
pub fn path_matches_filter(file_path: &str, filter: &str) -> bool {
    let filter_lower = filter.to_lowercase();
    if contains_case_insensitive(file_path, &filter_lower) {
        return true;
    }

    Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|name| contains_case_insensitive(name, &filter_lower))
        .unwrap_or(false)
}

