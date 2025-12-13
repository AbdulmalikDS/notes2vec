use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Parsing error: {0}")]
    Parsing(String),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("HuggingFace API error: {0}")]
    HuggingFace(#[from] hf_hub::api::sync::ApiError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

pub type Result<T> = std::result::Result<T, Error>;

