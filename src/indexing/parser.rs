use crate::core::error::Result;
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
    let content = std::fs::read_to_string(path)?;

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

/// Chunking configuration
const MIN_CHUNK_SIZE: usize = 50;  // Minimum characters per chunk
const MAX_CHUNK_SIZE: usize = 500; // Maximum characters per chunk
const TARGET_CHUNK_SIZE: usize = 300; // Target size for optimal embeddings

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
                        end_line: line_number.max(chunk_start_line),
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
                
                // Headings end with a newline
                line_number += 1;
                chunk_start_line = line_number;
            }
            Event::Text(text) => {
                // Count newlines in text (rare but possible in code blocks or pasted text)
                let newlines = text.chars().filter(|&c| c == '\n').count();
                if in_heading {
                    heading_text.push_str(text);
                } else {
                    current_text.push_str(text);
                    current_text.push(' ');
                }
                line_number += newlines;
            }
            Event::SoftBreak | Event::HardBreak => {
                if !in_heading {
                    current_text.push('\n');
                    line_number += 1;
                }
            }
            Event::End(TagEnd::Paragraph) => {
                // Paragraphs end with a newline (or two)
                line_number += 1;
                
                // If text exceeds max size, split intelligently at sentence boundaries
                if current_text.len() > MAX_CHUNK_SIZE {
                    let new_chunks = split_text_intelligently(
                        &current_text,
                        &header_stack,
                        chunk_start_line,
                        line_number - 1, // End line of the paragraph
                        &mut chunk_index,
                    );
                    chunks.extend(new_chunks);
                    current_text.clear();
                    chunk_start_line = line_number;
                }
            }
            Event::End(TagEnd::Item) => {
                line_number += 1;
            }
            Event::End(TagEnd::CodeBlock) => {
                line_number += 1;
            }
            _ => {}
        }
    }

    // Add remaining text as final chunk
    if !current_text.trim().is_empty() {
        // If remaining text is too large, split it
        if current_text.len() > MAX_CHUNK_SIZE {
            let new_chunks = split_text_intelligently(
                &current_text,
                &header_stack,
                chunk_start_line,
                line_number,
                &mut chunk_index,
            );
            chunks.extend(new_chunks);
        } else {
            chunks.push(TextChunk {
                text: current_text.trim().to_string(),
                context: build_context(&header_stack),
                chunk_index,
                start_line: chunk_start_line,
                end_line: line_number.max(chunk_start_line),
            });
        }
    }

    Ok((title, header_stack, chunks))
}

/// Split text intelligently at sentence boundaries while respecting size constraints
fn split_text_intelligently(
    text: &str,
    header_stack: &[String],
    start_line: usize,
    end_line: usize,
    chunk_index: &mut usize,
) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    let trimmed = text.trim();
    
    if trimmed.is_empty() {
        return chunks;
    }

    // Split by sentence boundaries (., !, ? followed by space, newline, or end)
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = trimmed.chars().collect();
    
    for (i, &ch) in chars.iter().enumerate() {
        if matches!(ch, '.' | '!' | '?') {
            // Check if followed by whitespace or end of string
            let next_char = chars.get(i + 1);
            if next_char.map(|c| c.is_whitespace()).unwrap_or(true) {
                // Use char_indices to get byte positions for slicing
                let byte_start = trimmed.char_indices().nth(start).map(|(pos, _)| pos).unwrap_or(0);
                let byte_end = trimmed.char_indices().nth(i).map(|(pos, _)| pos + 1).unwrap_or(trimmed.len());
                sentences.push(&trimmed[byte_start..byte_end.min(trimmed.len())]);
                start = i + 1;
            }
        }
    }
    
    // Add remaining text if any
    if start < chars.len() {
        let byte_start = trimmed.char_indices().nth(start).map(|(pos, _)| pos).unwrap_or(trimmed.len());
        sentences.push(&trimmed[byte_start..]);
    }

    let mut current_chunk = String::new();
    let mut current_start = start_line;
    let context = build_context(header_stack);

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        // If adding this sentence would exceed max size, save current chunk
        // Also try to target TARGET_CHUNK_SIZE for optimal embedding quality
        let would_exceed_max = !current_chunk.is_empty() 
            && current_chunk.len() + sentence.len() + 1 > MAX_CHUNK_SIZE;
        let reached_target = !current_chunk.is_empty()
            && current_chunk.len() >= TARGET_CHUNK_SIZE
            && current_chunk.len() + sentence.len() + 1 > MAX_CHUNK_SIZE;
        
        if (would_exceed_max || reached_target) && current_chunk.len() >= MIN_CHUNK_SIZE {
            chunks.push(TextChunk {
                text: current_chunk.trim().to_string(),
                context: context.clone(),
                chunk_index: *chunk_index,
                start_line: current_start,
                end_line: end_line,
            });
            *chunk_index += 1;
            current_chunk.clear();
            current_start = end_line;
        }

        // Add sentence to current chunk
        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(sentence);
    }

    // Add remaining chunk if it meets minimum size
    if !current_chunk.trim().is_empty() && current_chunk.len() >= MIN_CHUNK_SIZE {
        chunks.push(TextChunk {
            text: current_chunk.trim().to_string(),
            context,
            chunk_index: *chunk_index,
            start_line: current_start,
            end_line,
        });
        *chunk_index += 1;
    } else if !current_chunk.trim().is_empty() {
        // If too small, merge with previous chunk or add anyway
        if let Some(last_chunk) = chunks.last_mut() {
            last_chunk.text.push_str(" ");
            last_chunk.text.push_str(&current_chunk.trim());
            last_chunk.end_line = end_line;
        } else {
            chunks.push(TextChunk {
                text: current_chunk.trim().to_string(),
                context,
                chunk_index: *chunk_index,
                start_line: current_start,
                end_line,
            });
            *chunk_index += 1;
        }
    }

    chunks
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
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_simple_markdown() {
        let content = "# Title\n\nThis is some content.";
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, "Title");
        assert!(!doc.chunks.is_empty());
    }

    #[test]
    fn test_parse_markdown_without_title() {
        let content = "This is content without a title.";
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        // Should use filename as title
        assert_eq!(doc.title, "test");
    }

    #[test]
    fn test_parse_empty_file() {
        let content = "";
        let result = parse_markdown(content, Path::new("empty.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, "empty");
        // Empty file might have no chunks or one empty chunk
        assert!(doc.chunks.is_empty() || doc.chunks.iter().all(|c| c.text.trim().is_empty()));
    }

    #[test]
    fn test_parse_frontmatter() {
        let content = r#"---
title: Test Document
tags: [rust, testing]
custom_field: custom_value
---

# Main Title

Content here.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.metadata.title, Some("Test Document".to_string()));
        assert_eq!(doc.metadata.tags.len(), 2);
        assert!(doc.metadata.tags.contains(&"rust".to_string()));
        assert!(doc.metadata.tags.contains(&"testing".to_string()));
        assert_eq!(doc.metadata.custom.get("custom_field"), Some(&"custom_value".to_string()));
    }

    #[test]
    fn test_parse_frontmatter_comma_separated_tags() {
        let content = r#"---
title: Test
tags: rust, testing, cli
---

Content.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.metadata.tags.len(), 3);
    }

    #[test]
    fn test_parse_frontmatter_no_tags() {
        let content = r#"---
title: Test
---

Content.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert!(doc.metadata.tags.is_empty());
    }

    #[test]
    fn test_parse_header_hierarchy() {
        let content = r#"# Level 1

Content 1.

## Level 2

Content 2.

### Level 3

Content 3.

## Another Level 2

Content 4.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, "Level 1");
        assert!(!doc.header_hierarchy.is_empty());
    }

    #[test]
    fn test_parse_chunking() {
        let content = r#"# Title

First paragraph with some content.

Second paragraph with more content.

## Section

Third paragraph.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert!(!doc.chunks.is_empty());
        
        // Verify chunks have text
        for chunk in &doc.chunks {
            assert!(!chunk.text.trim().is_empty());
            assert!(!chunk.context.is_empty() || chunk.chunk_index == 0);
        }
    }

    #[test]
    fn test_parse_chunking_large_text() {
        // Create text that exceeds MAX_CHUNK_SIZE
        let mut content = "# Title\n\n".to_string();
        let large_paragraph = "This is a sentence. ".repeat(100); // ~2000 characters
        content.push_str(&large_paragraph);
        
        let result = parse_markdown(&content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        
        // Should be split into multiple chunks
        assert!(doc.chunks.len() > 1);
        
        // Each chunk should be within size limits
        for chunk in &doc.chunks {
            assert!(chunk.text.len() <= MAX_CHUNK_SIZE);
        }
    }

    #[test]
    fn test_parse_chunk_context() {
        let content = r#"# Document

Content at root.

## Section 1

Content in section 1.

### Subsection 1.1

Content in subsection.

## Section 2

Content in section 2.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        
        // Verify chunks have appropriate context
        for chunk in &doc.chunks {
            if chunk.context.contains("Section 1") {
                assert!(chunk.context.contains("Document"));
            }
            if chunk.context.contains("Subsection 1.1") {
                assert!(chunk.context.contains("Section 1"));
            }
        }
    }

    #[test]
    fn test_parse_chunk_line_numbers() {
        let content = r#"# Title

Line 3 content.

Line 5 content.

## Section

Line 9 content.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        
        // Verify chunks have line numbers
        for chunk in &doc.chunks {
            assert!(chunk.start_line > 0);
            assert!(chunk.end_line >= chunk.start_line);
        }
    }

    #[test]
    fn test_parse_markdown_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.md");
        
        let content = r#"# Test Document

This is test content.
"#;
        fs::write(&test_file, content).unwrap();
        
        let result = parse_markdown_file(&test_file);
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert_eq!(doc.title, "Test Document");
        assert!(!doc.chunks.is_empty());
    }

    #[test]
    fn test_parse_markdown_file_nonexistent() {
        let result = parse_markdown_file(Path::new("/nonexistent/file.md"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_multiple_h1_headers() {
        let content = r#"# First Title

Content 1.

# Second Title

Content 2.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        // First H1 should be title
        assert_eq!(doc.title, "First Title");
    }

    #[test]
    fn test_parse_code_blocks() {
        let content = r#"# Title

Here is some code:

```rust
fn main() {
    println!("Hello");
}
```

More content.
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        // Code blocks should be included in chunks
        assert!(!doc.chunks.is_empty());
    }

    #[test]
    fn test_parse_lists() {
        let content = r#"# Title

- Item 1
- Item 2
- Item 3

1. Numbered 1
2. Numbered 2
"#;
        let result = parse_markdown(content, Path::new("test.md"));
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert!(!doc.chunks.is_empty());
    }

    #[test]
    fn test_build_context() {
        let headers = vec!["Document".to_string(), "Section".to_string(), "Subsection".to_string()];
        let context = build_context(&headers);
        assert_eq!(context, "Document > Section > Subsection");
    }

    #[test]
    fn test_build_context_empty() {
        let headers = vec![];
        let context = build_context(&headers);
        assert_eq!(context, "");
    }

    #[test]
    fn test_build_context_single() {
        let headers = vec!["Document".to_string()];
        let context = build_context(&headers);
        assert_eq!(context, "Document");
    }
}

