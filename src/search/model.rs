use crate::core::config::Config;
use crate::core::error::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::sync::Api;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, Tokenizer};

/// Default embedding model (small, fast, good for semantic search)
const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Embedding model manager
pub struct EmbeddingModel {
    model: Option<Arc<Mutex<BertModel>>>,
    tokenizer: Option<Arc<Mutex<Tokenizer>>>,
    device: Device,
    embedding_dim: usize,
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    tokenizer_path: PathBuf,
}

impl EmbeddingModel {
    /// Initialize model manager
    pub fn init(config: &Config) -> Result<Self> {
        // Ensure models directory exists
        std::fs::create_dir_all(&config.models_dir)?;

        let model_path = config.models_dir.join("model.safetensors");
        let config_path = config.models_dir.join("config.json");
        let tokenizer_path = config.models_dir.join("tokenizer.json");

        // Try to download and load model if files don't exist
        let (model, tokenizer, embedding_dim) = if model_path.exists() && config_path.exists() && tokenizer_path.exists() {
            // Load existing model files
            Self::load_model_files(&model_path, &config_path, &tokenizer_path)?
        } else {
            // Try to download model from HuggingFace
            match Self::download_model(config, &model_path, &config_path, &tokenizer_path) {
                Ok((m, t, dim)) => (m, t, dim),
                Err(e) => {
                    // Download failed - use fallback
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        eprintln!("⚠ Could not download model: {}", e);
                        eprintln!("  Using hash-based embeddings as fallback.");
                        eprintln!("  To use full model, ensure network connectivity and try again.");
                    });
                    (None, None, 384) // Default dimension for MiniLM
                }
            }
        };

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_dim,
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
    ) -> Result<(Option<Arc<Mutex<BertModel>>>, Option<Arc<Mutex<Tokenizer>>>, usize)> {
        println!("Downloading embedding model from HuggingFace Hub...");
        println!("Model: {}", DEFAULT_MODEL);
        
        // Initialize API
        let api = Api::new().map_err(|e| {
            Error::HuggingFace(e)
        })?;
        
        // Get model repository
        let repo = api.model(DEFAULT_MODEL.to_string());
        
        // Download required files
        println!("  Downloading config.json...");
        let config_file = repo.get("config.json")
            .map_err(|e| Error::HuggingFace(e))?;
        
        println!("  Downloading tokenizer.json...");
        let tokenizer_file = repo.get("tokenizer.json")
            .map_err(|e| Error::HuggingFace(e))?;
        
        println!("  Downloading model.safetensors (this may take a while)...");
        let weights_file = repo.get("model.safetensors")
            .map_err(|e| Error::HuggingFace(e))?;

        // Copy files to our models directory
        std::fs::copy(&config_file, config_path)?;
        std::fs::copy(&tokenizer_file, tokenizer_path)?;
        std::fs::copy(&weights_file, model_path)?;

        println!("✓ Model downloaded successfully");

        // Load the downloaded model
        Self::load_model_files(model_path, config_path, tokenizer_path)
    }

    /// Load model files from disk
    fn load_model_files(
        model_path: &PathBuf,
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
    ) -> Result<(Option<Arc<Mutex<BertModel>>>, Option<Arc<Mutex<Tokenizer>>>, usize)> {
        println!("Loading model from disk...");
        
        // Load and parse config
        let config_content = std::fs::read_to_string(config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| Error::Model(format!("Failed to parse config: {}", e)))?;

        // Default embedding dimension for MiniLM-L6-v2
        let embedding_dim = 384;

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

        println!("✓ Model loaded successfully");

        Ok((
            Some(Arc::new(Mutex::new(model))),
            Some(Arc::new(Mutex::new(tokenizer))),
            embedding_dim,
        ))
    }

    /// Check if full model is available
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some() && self.tokenizer.is_some()
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Generate embeddings for texts
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if let (Some(model), Some(tokenizer)) = (&self.model, &self.tokenizer) {
            self.embed_with_model(model, tokenizer, texts)
        } else {
            // Fallback: Use hash-based embeddings
            self.embed_with_hash(texts)
        }
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

        // Convert token IDs and attention masks to tensors
        let token_ids: Result<Vec<Tensor>> = tokens
            .iter()
            .map(|t| {
                let ids: Vec<u32> = t.get_ids().iter().map(|&id| id as u32).collect();
                Tensor::new(ids.as_slice(), &self.device).map_err(Error::Candle)
            })
            .collect();

        let attention_masks: Result<Vec<Tensor>> = tokens
            .iter()
            .map(|t| {
                let mask: Vec<u32> = t.get_attention_mask().iter().map(|&m| m as u32).collect();
                Tensor::new(mask.as_slice(), &self.device).map_err(Error::Candle)
            })
            .collect();

        let token_ids = Tensor::stack(&token_ids?, 0)?;
        let attention_mask = Tensor::stack(&attention_masks?, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Forward pass through BERT model
        let embeddings = model_guard.forward(&token_ids, &token_type_ids)?;

        // Mean pooling with attention mask
        let attention_mask_3d = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = attention_mask_3d.sum(1)?;
        let masked_embeddings = embeddings.broadcast_mul(&attention_mask_3d)?.sum(1)?;
        let pooled = masked_embeddings.broadcast_div(&sum_mask)?;

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

    /// Generate hash-based embeddings (fallback when model unavailable)
    fn embed_with_hash(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let mut embedding = vec![0.0f32; self.embedding_dim];
            
            // Generate SHA256 hash
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            let hash = hasher.finalize();

            // Distribute hash bytes across embedding dimensions
            for (i, &byte) in hash.iter().enumerate() {
                let idx = i % self.embedding_dim;
                embedding[idx] = (byte as f32 / 255.0) * 2.0 - 1.0;
            }

            // L2 normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                embedding.iter_mut().for_each(|val| *val /= norm);
            }

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}
