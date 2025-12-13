# notes2vec

A lightweight, local-first semantic search engine for personal notes, journals, and documentation.

![notes2vec in action](assets/notes2vec.gif)

## Installation

Download pre-built binaries from the [Releases page](https://github.com/AbdulmalikDS/notes2vec/releases).

**Windows:**
```bash
# Download notes2vec-x86_64-pc-windows-msvc.zip
# Extract and add to PATH, or run directly
```

**macOS:**
```bash
# Download notes2vec-x86_64-apple-darwin.tar.gz (Intel)
# or notes2vec-aarch64-apple-darwin.tar.gz (Apple Silicon)
# Extract and move to /usr/local/bin/ or add to PATH
```

**Linux:**
```bash
# Download notes2vec-x86_64-unknown-linux-gnu.tar.gz
# Extract and move to /usr/local/bin/ or add to PATH
```

## Quick Start

```bash
# 1. Initialize (downloads embedding model, ~80MB)
notes2vec init

# 2. Index your notes
notes2vec index /path/to/notes

# 3. Search - TUI opens automatically
notes2vec
```

## Building from Source

```bash
git clone https://github.com/AbdulmalikDS/notes2vec.git
cd notes2vec
cargo build --release
```

## License

MIT
