// Core functionality
pub mod core {
    pub mod config;
    pub mod error;
}

// Data storage
pub mod storage {
    pub mod state;
    pub mod vectors;
}

// Indexing pipeline
pub mod indexing {
    pub mod discovery;
    pub mod parser;
}

// Search & ML
pub mod search {
    pub mod model;
}

// User interfaces
pub mod ui {
    pub mod cli;
    pub mod tui;
    pub mod watch;
}

// Re-export commonly used types
pub use core::error::{Error, Result};
pub use core::config::Config;
pub use storage::state::{StateStore, calculate_file_hash, get_file_modified_time};
pub use storage::vectors::{VectorStore, VectorEntry};
pub use indexing::discovery::discover_files;
pub use indexing::parser;
pub use search::model::EmbeddingModel;
pub use ui::cli::Cli;
pub use ui::tui::SearchTui;
pub use ui::watch::FileWatcher;

