#!/usr/bin/env python3
"""
Model Manager for AZR system.
Handles model downloading, caching, and loading from local storage.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import shutil

class ModelManager:
    """Manages model downloads and caching for the AZR system."""
    
    def __init__(self, cache_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save model metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get the cache path for a model."""
        # Replace slashes with underscores for filesystem compatibility
        safe_name = model_name.replace('/', '_')
        return self.cache_dir / safe_name
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is cached locally."""
        cache_path = self._get_cache_path(model_name)
        return cache_path.exists() and model_name in self.metadata
    
    def download_and_cache_model(self, model_name: str) -> Tuple[Path, Dict[str, Any]]:
        """
        Download and cache a model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Tuple of (cache_path, model_info)
        """
        cache_path = self._get_cache_path(model_name)
        
        self.logger.info(f"Downloading model {model_name} to {cache_path}")
        
        # Download model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU optimized
                low_cpu_mem_usage=True
            )
            
            # Save to cache
            cache_path.mkdir(exist_ok=True)
            tokenizer.save_pretrained(cache_path)
            model.save_pretrained(cache_path)
            
            # Calculate model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)  # float32
            
            # Update metadata
            model_info = {
                'model_name': model_name,
                'cache_path': str(cache_path),
                'downloaded_at': datetime.now().isoformat(),
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'model_type': model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
            }
            
            self.metadata[model_name] = model_info
            self._save_metadata()
            
            self.logger.info(f"Model cached successfully: {model_size_mb:.1f} MB")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return cache_path, model_info
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            raise
    
    def load_cached_model(self, model_name: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load a cached model.
        
        Args:
            model_name: Model name
            
        Returns:
            Tuple of (tokenizer, model, model_info)
        """
        if not self.is_model_cached(model_name):
            raise ValueError(f"Model {model_name} is not cached. Download it first.")
        
        cache_path = self._get_cache_path(model_name)
        model_info = self.metadata[model_name]
        
        self.logger.info(f"Loading cached model from {cache_path}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(cache_path)
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        return tokenizer, model, model_info
    
    def get_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all cached models."""
        return self.metadata.copy()
    
    def get_cache_size(self) -> Dict[str, float]:
        """Get total cache size."""
        total_size = 0
        for model_info in self.metadata.values():
            cache_path = Path(model_info['cache_path'])
            if cache_path.exists():
                # Calculate directory size
                for path in cache_path.rglob('*'):
                    if path.is_file():
                        total_size += path.stat().st_size
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024)
        }
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to remove, or None to clear all
        """
        if model_name:
            cache_path = self._get_cache_path(model_name)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                self.logger.info(f"Removed cached model: {model_name}")
            
            if model_name in self.metadata:
                del self.metadata[model_name]
                self._save_metadata()
        else:
            # Clear all cache
            for path in self.cache_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
            self.metadata = {}
            self._save_metadata()
            self.logger.info("Cleared all model cache")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return self.metadata.get(model_name)


# Singleton instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def ensure_model_cached(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Ensure a model is cached, downloading if necessary.
    
    Args:
        model_name: Model to cache
        
    Returns:
        Model information dictionary
    """
    manager = get_model_manager()
    
    if manager.is_model_cached(model_name):
        return manager.get_model_info(model_name)
    else:
        try:
            _, model_info = manager.download_and_cache_model(model_name)
            return model_info
        except Exception as e:
            logging.error(f"Failed to cache model {model_name}: {e}")
            return None
