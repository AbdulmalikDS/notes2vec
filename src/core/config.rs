use super::error::{Error, Result};
use std::path::PathBuf;

/// Configuration for notes2vec
#[derive(Debug, Clone)]
pub struct Config {
    /// Base directory for notes2vec data
    pub base_dir: PathBuf,
    /// Directory for the vector database
    pub database_dir: PathBuf,
    /// Directory for cached models
    pub models_dir: PathBuf,
    /// Path to the state store
    pub state_path: PathBuf,
}

impl Config {
    /// Get the default configuration directory
    pub fn default_base_dir() -> Result<PathBuf> {
        dirs::home_dir()
            .ok_or_else(|| Error::Config("Could not determine home directory".to_string()))
            .map(|home| home.join(".notes2vec"))
    }

    /// Create a new configuration
    pub fn new(base_dir: Option<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.unwrap_or_else(|| {
            Self::default_base_dir().unwrap_or_else(|_| PathBuf::from(".notes2vec"))
        });

        Ok(Self {
            database_dir: base_dir.join("database"),
            models_dir: base_dir.join("models"),
            state_path: base_dir.join("state").join("state.redb"),
            base_dir,
        })
    }

    /// Initialize the configuration directories
    pub fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        std::fs::create_dir_all(&self.database_dir)?;
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(self.state_path.parent().unwrap())?;
        Ok(())
    }

    /// Check if the configuration is already initialized
    pub fn is_initialized(&self) -> bool {
        self.base_dir.exists() && self.database_dir.exists()
    }
}

