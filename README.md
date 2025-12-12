# notes2vec

A lightweight, local-first semantic search engine for personal notes, journals, and documentation.

## Overview

notes2vec helps you find your personal notes based on meaning, not just keywords. Unlike traditional search tools that require exact word matches, notes2vec understands what you're looking for even if you use different words than what you wrote.

## Features

- **Local-First**: All processing happens on your machine. No cloud, no external services.
- **Semantic Search**: Find notes by meaning, not just keywords.
- **Automatic Indexing**: Watches your note directories and updates the index automatically.
- **Privacy-Focused**: Your data never leaves your computer.
- **Zero Dependencies**: Single binary, no Rust installation needed, no Docker or external services required.

## Installation

**No Rust Required!** notes2vec is distributed as a standalone binary. You don't need to install Rust or compile anything.

Download pre-built binaries for Windows, macOS, and Linux from the [Releases page](https://github.com/AbdulmalikDS/notes2vec/releases).

**Windows:**
- Download `notes2vec-x86_64-pc-windows-msvc.zip`
- Extract and add to your PATH, or run directly

**macOS:**
- Download `notes2vec-x86_64-apple-darwin.tar.gz` (Intel) or `notes2vec-aarch64-apple-darwin.tar.gz` (Apple Silicon)
- Extract and move to `/usr/local/bin/` or add to your PATH

**Linux:**
- Download `notes2vec-x86_64-unknown-linux-gnu.tar.gz`
- Extract and move to `/usr/local/bin/` or add to your PATH

## Quick Start

### 1. Initialize

```bash
notes2vec init
```

This sets up the local database and downloads the embedding model (one-time operation).

### 2. Index Your Notes

```bash
notes2vec index /path/to/notes
```

### 3. Search

```bash
notes2vec search "how to configure database"
```

### 4. Watch for Changes (Daemon Mode)

```bash
notes2vec watch /path/to/notes
```

This continuously monitors your notes and automatically updates the index when files change.

## Technology Stack

- **Language**: Rust
- **ML Framework**: Candle (pure Rust, quantized models)
- **Embedding Model**: nomic-embed-text-v1.5 (quantized, 8-bit)
- **Vector Database**: LanceDB (embedded, local, free)
- **State Management**: redb (lightweight key-value store)

## Project Status

**Early Development** - This project is currently in active development.

## License

Licensed under the MIT license 
