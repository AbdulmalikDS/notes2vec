use crate::error::{Error, Result};
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use std::path::Path;

/// Metadata extracted from frontmatter
#[derive(Debug, Clone, Default)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub created: Option<String>,
    pub modified: Option<String>,
    pub custom: std::collections::HashMap<String, String>,
}

/// A chunk of text with its context
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub context: String, // e.g., "Document Title > Section > Subsection"
    pub chunk_index: usize,
    pub start_line: usize,
    pub end_line: usize,
}

/// Parsed document structure
#[derive(Debug, Clone)]
pub struct ParsedDocument {
    pub metadata: DocumentMetadata,
    pub title: String,
    pub chunks: Vec<TextChunk>,
    pub header_hierarchy: Vec<String>,
}

/// Parse a Markdown file and extract structure
pub fn parse_markdown_file(path: &Path) -> Result<ParsedDocument> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(e))?;

    parse_markdown(&content, path)
}

/// Parse Markdown content
pub fn parse_markdown(content: &str, path: &Path) -> Result<ParsedDocument> {
    // Extract frontmatter
    let (frontmatter, markdown_content) = extract_frontmatter(content);
    let metadata = parse_frontmatter(frontmatter);

    // Parse Markdown structure
    let (title, header_hierarchy, chunks) = parse_structure(&markdown_content)?;

    Ok(ParsedDocument {
        metadata,
        title: title.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled")
                .to_string()
        }),
        chunks,
        header_hierarchy,
    })
}

/// Extract frontmatter from content
fn extract_frontmatter(content: &str) -> (Option<String>, String) {
    // Simple frontmatter extraction - look for YAML between --- markers
    if content.starts_with("---\n") {
        if let Some(end_pos) = content[4..].find("\n---\n") {
            let frontmatter = content[4..end_pos + 4].to_string();
            let markdown_content = content[end_pos + 9..].to_string();
            return (Some(frontmatter), markdown_content);
        }
    }
    (None, content.to_string())
}

/// Parse frontmatter YAML into metadata
fn parse_frontmatter(frontmatter: Option<String>) -> DocumentMetadata {
    let mut metadata = DocumentMetadata::default();

    if let Some(fm) = frontmatter {
        if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(&fm) {
            if let Some(map) = value.as_mapping() {
                // Extract common fields
                if let Some(title) = map.get("title").and_then(|v| v.as_str()) {
                    metadata.title = Some(title.to_string());
                }

                if let Some(tags) = map.get("tags") {
                    if let Some(tag_array) = tags.as_sequence() {
                        metadata.tags = tag_array
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                    } else if let Some(tag_str) = tags.as_str() {
                        // Handle comma-separated tags
                        metadata.tags = tag_str
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                }

                // Extract custom fields
                for (key, value) in map.iter() {
                    if let (Some(k), Some(v)) = (key.as_str(), value.as_str()) {
                        if !matches!(k, "title" | "tags" | "created" | "modified") {
                            metadata.custom.insert(k.to_string(), v.to_string());
                        }
                    }
                }
            }
        }
    }

    metadata
}

/// Parse Markdown structure and extract chunks
fn parse_structure(content: &str) -> Result<(Option<String>, Vec<String>, Vec<TextChunk>)> {
    let parser = Parser::new(content);
    let events: Vec<Event> = parser.collect();

    let mut title: Option<String> = None;
    let mut header_stack: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut chunks = Vec::new();
    let mut chunk_index = 0;
    let mut line_number = 1;
    let mut chunk_start_line = 1;
    let mut in_heading = false;
    let mut current_heading_level = 0;
    let mut heading_text = String::new();

    for event in &events {
        match event {
            Event::Start(Tag::Heading { level, id: _, classes: _, attrs: _ }) => {
                // Save current chunk if we have text
                if !current_text.trim().is_empty() {
                    chunks.push(TextChunk {
                        text: current_text.trim().to_string(),
                        context: build_context(&header_stack),
                        chunk_index,
                        start_line: chunk_start_line,
                        end_line: line_number - 1,
                    });
                    chunk_index += 1;
                    current_text.clear();
                }

                in_heading = true;
                current_heading_level = *level as usize;
                heading_text.clear();
                chunk_start_line = line_number;
            }
            Event::End(TagEnd::Heading(_)) if in_heading => {
                in_heading = false;
                let heading = heading_text.trim().to_string();

                // Update header stack using the level we captured
                let level = current_heading_level;
                header_stack.truncate(level.saturating_sub(1));
                header_stack.push(heading.clone());

                // First H1 becomes the title
                if level == 1 && title.is_none() {
                    title = Some(heading.clone());
                }
            }
            Event::Text(text) => {
                if in_heading {
                    heading_text.push_str(text);
                } else {
                    current_text.push_str(text);
                    current_text.push(' ');
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if !in_heading {
                    current_text.push('\n');
                    line_number += 1;
                }
            }
            Event::End(TagEnd::Paragraph) => {
                // Simple chunking: split on paragraphs for now
                // TODO: Implement semantic chunking with semchunk-rs
                if current_text.len() > 500 {
                    // Split large paragraphs
                    let words: Vec<&str> = current_text.split_whitespace().collect();
                    let chunk_size = 100; // words per chunk

                    for chunk_words in words.chunks(chunk_size) {
                        let chunk_text = chunk_words.join(" ");
                        if !chunk_text.trim().is_empty() {
                            chunks.push(TextChunk {
                                text: chunk_text.trim().to_string(),
                                context: build_context(&header_stack),
                                chunk_index,
                                start_line: chunk_start_line,
                                end_line: line_number,
                            });
                            chunk_index += 1;
                        }
                    }
                    current_text.clear();
                    chunk_start_line = line_number + 1;
                }
            }
            _ => {}
        }
    }

    // Add remaining text as final chunk
    if !current_text.trim().is_empty() {
        chunks.push(TextChunk {
            text: current_text.trim().to_string(),
            context: build_context(&header_stack),
            chunk_index,
            start_line: chunk_start_line,
            end_line: line_number,
        });
    }

    Ok((title, header_stack, chunks))
}

/// Build context string from header hierarchy
fn build_context(headers: &[String]) -> String {
    if headers.is_empty() {
        return String::new();
    }
    headers.join(" > ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_markdown() {
        let content = "# Title\n\nThis is some content.";
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, "Title");
        assert!(!doc.chunks.is_empty());
    }
}

