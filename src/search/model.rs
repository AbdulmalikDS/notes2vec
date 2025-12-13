use crate::core::config::Config;
use crate::core::error::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::sync::Api;
// sha2 dependency is used elsewhere; no hashing fallback is used for embeddings.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, Tokenizer};

/// Default embedding model (small, strong, good for semantic search)
/// https://huggingface.co/BAAI/bge-small-en-v1.5
const DEFAULT_MODEL: &str = "BAAI/bge-small-en-v1.5";

/// Identifier for the embedding model used to build/query the index.
/// If this changes, you should re-index.
pub const EMBEDDING_MODEL_ID: &str = DEFAULT_MODEL;

/// Embedding model manager
pub struct EmbeddingModel {
    model: Option<Arc<Mutex<BertModel>>>,
    tokenizer: Option<Arc<Mutex<Tokenizer>>>,
    device: Device,
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    tokenizer_path: PathBuf,
}

impl EmbeddingModel {
    /// Initialize model manager
    pub fn init(config: &Config) -> Result<Self> {
        // Default to quiet so TUI runs cleanly (no stray stdout/stderr output).
        Self::init_with_verbosity(config, false)
    }

    /// Initialize model manager without printing to stdout/stderr.
    /// Useful for TUI mode (raw/alternate screen) to avoid corrupting the UI.
    pub fn init_quiet(config: &Config) -> Result<Self> {
        Self::init_with_verbosity(config, false)
    }

    /// Initialize model manager with progress/status output.
    /// Useful for non-TUI CLI commands where stdout is expected.
    pub fn init_verbose(config: &Config) -> Result<Self> {
        Self::init_with_verbosity(config, true)
    }

    fn init_with_verbosity(config: &Config, verbose: bool) -> Result<Self> {
        // Ensure models directory exists
        std::fs::create_dir_all(&config.models_dir)?;

        let model_path = config.models_dir.join("model.safetensors");
        let config_path = config.models_dir.join("config.json");
        let tokenizer_path = config.models_dir.join("tokenizer.json");

        // Try to download and load model if files don't exist.
        // No fallback: if the model can't be loaded, return an error.
        let (model, tokenizer) = if model_path.exists() && config_path.exists() && tokenizer_path.exists() {
            Self::load_model_files(&model_path, &config_path, &tokenizer_path, verbose)?
        } else {
            Self::download_model(config, &model_path, &config_path, &tokenizer_path, verbose)?
        };

        if model.is_none() || tokenizer.is_none() {
            return Err(Error::Model(
                "Embedding model not available (no fallback). Run `notes2vec init` to download it.".to_string(),
            ));
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        Ok(Self {
            model,
            tokenizer,
            device,
            model_path,
            tokenizer_path,
        })
    }

    /// Download model from HuggingFace Hub
    fn download_model(
        _config: &Config,
        model_path: &PathBuf,
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
        verbose: bool,
    ) -> Result<(Option<Arc<Mutex<BertModel>>>, Option<Arc<Mutex<Tokenizer>>>)> {
        if verbose {
            println!("Downloading embedding model from HuggingFace Hub...");
            println!("Model: {}", DEFAULT_MODEL);
        }
        
        // Initialize API
        let api = Api::new().map_err(|e| {
            Error::HuggingFace(e)
        })?;
        
        // Get model repository
        let repo = api.model(DEFAULT_MODEL.to_string());
        
        // Download required files
        if verbose {
            println!("  Downloading config.json...");
        }
        let config_file = repo.get("config.json")
            .map_err(|e| Error::HuggingFace(e))?;
        
        if verbose {
            println!("  Downloading tokenizer.json...");
        }
        let tokenizer_file = repo.get("tokenizer.json")
            .map_err(|e| Error::HuggingFace(e))?;
        
        if verbose {
            println!("  Downloading model.safetensors (this may take a while)...");
        }
        let weights_file = repo.get("model.safetensors")
            .map_err(|e| Error::HuggingFace(e))?;

        // Copy files to our models directory
        std::fs::copy(&config_file, config_path)?;
        std::fs::copy(&tokenizer_file, tokenizer_path)?;
        std::fs::copy(&weights_file, model_path)?;

        if verbose {
            println!("✓ Model downloaded successfully");
        }

        // Load the downloaded model
        Self::load_model_files(model_path, config_path, tokenizer_path, verbose)
    }

    /// Load model files from disk
    fn load_model_files(
        model_path: &PathBuf,
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
        verbose: bool,
    ) -> Result<(Option<Arc<Mutex<BertModel>>>, Option<Arc<Mutex<Tokenizer>>>)> {
        if verbose {
            println!("Loading model from disk...");
        }
        
        // Load and parse config
        let config_content = std::fs::read_to_string(config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| Error::Model(format!("Failed to parse config: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // Determine device (CUDA if available, else CPU)
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        // Load model weights using memory mapping
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
                .map_err(|e| Error::Model(format!("Failed to load weights: {}", e)))?
        };

        // Load BERT model
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| Error::Model(format!("Failed to load model: {}", e)))?;

        if verbose {
            println!("✓ Model loaded successfully");
        }

        Ok((
            Some(Arc::new(Mutex::new(model))),
            Some(Arc::new(Mutex::new(tokenizer))),
        ))
    }

    /// Check if full model is available
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some() && self.tokenizer.is_some()
    }

    /// Generate embeddings for texts
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if let (Some(model), Some(tokenizer)) = (&self.model, &self.tokenizer) {
            self.embed_with_model(model, tokenizer, texts)
        } else {
            Err(Error::Model(
                "Embedding model not loaded (no fallback). Run `notes2vec init`.".to_string(),
            ))
        }
    }

    /// Embed query texts (recommended for BGE models).
    pub fn embed_queries(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts.iter().map(|t| format!("query: {}", t)).collect();
        self.embed(&prefixed)
    }

    /// Embed passage texts (recommended for BGE models).
    pub fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts.iter().map(|t| format!("passage: {}", t)).collect();
        self.embed(&prefixed)
    }

    /// Generate embeddings using the loaded BERT model
    fn embed_with_model(
        &self,
        model: &Arc<Mutex<BertModel>>,
        tokenizer: &Arc<Mutex<Tokenizer>>,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        // Lock model and tokenizer
        let model_guard = model.lock()
            .map_err(|e| Error::Model(format!("Failed to lock model: {}", e)))?;
        let mut tokenizer_guard = tokenizer.lock()
            .map_err(|e| Error::Model(format!("Failed to lock tokenizer: {}", e)))?;

        // Configure tokenizer padding for batch processing
        if let Some(pp) = tokenizer_guard.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer_guard.with_padding(Some(pp));
        }

        // Tokenize texts
        let tokens = tokenizer_guard
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        // Convert token IDs to tensors
        let token_ids: Result<Vec<Tensor>> = tokens
            .iter()
            .map(|t| {
                let ids: Vec<u32> = t.get_ids().iter().map(|&id| id as u32).collect();
                Tensor::new(ids.as_slice(), &self.device).map_err(Error::Candle)
            })
            .collect();

        let token_ids = Tensor::stack(&token_ids?, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Forward pass through BERT model
        let embeddings = model_guard.forward(&token_ids, &token_type_ids)?;

        // CLS pooling (recommended for BGE-style retrieval models)
        // embeddings: [batch, seq, hidden] -> pooled: [batch, hidden]
        let pooled = embeddings.narrow(1, 0, 1)?.squeeze(1)?;

        // L2 normalization
        let normalized = Self::normalize_l2(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let (n_sentences, _) = normalized.dims2()?;
        let mut result = Vec::with_capacity(n_sentences);
        
        for i in 0..n_sentences {
            let embedding = normalized.get(i)?;
            result.push(embedding.to_vec1()?);
        }

        Ok(result)
    }

    /// L2 normalization
    fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    // (Hash-based fallback removed intentionally)
}
