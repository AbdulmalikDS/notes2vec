use clap::{Parser, Subcommand};

/// notes2vec - Local semantic search for personal notes
#[derive(Parser, Debug)]
#[command(name = "notes2vec")]
#[command(about = "A lightweight, local-first semantic search engine for personal notes", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize notes2vec (create directories and download model)
    Init {
        /// Custom base directory (default: ~/.notes2vec)
        #[arg(short, long)]
        base_dir: Option<String>,
    },
    /// Index notes from a directory
    Index {
        /// Path to the notes directory
        path: String,
        /// Force re-indexing of all files
        #[arg(short, long)]
        force: bool,
        /// Custom base directory (default: ~/.notes2vec)
        #[arg(long)]
        base_dir: Option<String>,
    },
    /// Watch a directory for changes and automatically update index
    Watch {
        /// Path to the notes directory
        path: String,
        /// Custom base directory (default: ~/.notes2vec)
        #[arg(long)]
        base_dir: Option<String>,
    },
    /// Search your notes
    Search {
        /// Search query (leave empty for interactive mode)
        query: Option<String>,
        /// Maximum number of results to return
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        /// Custom base directory (default: ~/.notes2vec)
        #[arg(long)]
        base_dir: Option<String>,
        /// Use interactive TUI mode
        #[arg(short, long)]
        interactive: bool,
    },
}

