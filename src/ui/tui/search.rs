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
    // Optimized: Pre-compute lowercase versions once to avoid repeated allocations
    if !q_lower.is_empty() {
        for (entry, sim) in results.iter_mut() {
            let mut bonus = 0.0f32;
            // Use case-insensitive contains check without allocating new strings
            if entry.file_path.to_lowercase().contains(&q_lower) {
                bonus += LEXICAL_BOOST_PATH;
            }
            if entry.context.to_lowercase().contains(&q_lower) {
                bonus += LEXICAL_BOOST_CONTEXT;
            }
            if entry.text.to_lowercase().contains(&q_lower) {
                bonus += LEXICAL_BOOST_TEXT;
            }
            *sim = (*sim + bonus).min(1.0);
        }
    }

    // Dedupe: keep best match per file, then take top results.
    // Optimized: Use references where possible to avoid unnecessary clones
    let mut best_by_file: HashMap<String, (VectorEntry, f32)> = HashMap::with_capacity(results.len());
    for (entry, sim) in results {
        best_by_file
            .entry(entry.file_path.clone())
            .and_modify(|current| {
                if sim > current.1 {
                    *current = (entry.clone(), sim);
                }
            })
            .or_insert_with(|| (entry, sim));
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

/// Check if a file path matches a filter string
pub fn path_matches_filter(file_path: &str, filter: &str) -> bool {
    let filter = filter.to_lowercase();
    let path_lower = file_path.to_lowercase();
    if path_lower.contains(&filter) {
        return true;
    }

    Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|name| name.to_lowercase().contains(&filter))
        .unwrap_or(false)
}

