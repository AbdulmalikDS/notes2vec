use crate::core::error::{Error, Result};
use crate::search::model::EmbeddingModel;
use crate::storage::vectors::{VectorEntry, VectorStore};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// Search configuration constants
const SEARCH_CANDIDATES_LIMIT: usize = 50;      // Number of candidates to fetch for unconstrained search
const SCOPED_SEARCH_CANDIDATES_LIMIT: usize = 200; // Number of candidates for scoped search
const MAX_RESULTS_DISPLAYED: usize = 5;         // Maximum number of results to display

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
    let mut results = if active_files.is_empty() {
        vector_store.search(query_embedding, SEARCH_CANDIDATES_LIMIT)?
    } else {
        vector_store.search_scoped(query_embedding, SCOPED_SEARCH_CANDIDATES_LIMIT, active_files)?
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

    // Dedupe: keep best match per file, then take top results.
    // Optimized: Avoid unnecessary clones by only cloning when updating
    let mut best_by_file: HashMap<String, (VectorEntry, f32)> = HashMap::with_capacity(results.len());
    for (entry, sim) in results {
        match best_by_file.get_mut(&entry.file_path) {
            Some(current) => {
                if sim > current.1 {
                    *current = (entry, sim);
                }
            }
            None => {
                best_by_file.insert(entry.file_path.clone(), (entry, sim));
            }
        }
    }
    let mut deduped: Vec<(VectorEntry, f32)> = best_by_file.into_values().collect();
    deduped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    deduped.truncate(MAX_RESULTS_DISPLAYED);

    Ok(deduped)
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

