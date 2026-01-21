"""Configuration management utilities.

This module provides utilities for loading and managing YAML configuration files.
"""

from typing import Any, Dict, Optional
import yaml
from pathlib import Path
import os


class Config:
    """Configuration manager for DTI prediction.
    
    Loads and manages configuration from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file (optional)
        """
        self.config = {}
        
        if config_path is not None:
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def save(self, config_path: str):
        """Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict):
        """Update configuration with dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively update nested dictionary.
        
        Args:
            base: Base dictionary to update
            updates: Dictionary with updates
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict:
        """Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using bracket notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.set(key, value)


def load_config(config_path: str, override: Optional[Dict] = None) -> Config:
    """Load configuration from file with optional overrides.
    
    Args:
        config_path: Path to configuration YAML file
        override: Optional dictionary of values to override
        
    Returns:
        Config object
    """
    config = Config(config_path)
    
    if override is not None:
        config.update(override)
    
    return config


def merge_configs(*config_paths: str) -> Config:
    """Merge multiple configuration files.
    
    Later configurations override earlier ones.
    
    Args:
        *config_paths: Paths to configuration files
        
    Returns:
        Merged configuration
    """
    merged = Config()
    
    for path in config_paths:
        with open(path, 'r') as f:
            updates = yaml.safe_load(f)
            merged.update(updates)
    
    return merged


def create_default_config() -> Dict:
    """Create default configuration dictionary.
    
    Returns:
        Default configuration
    """
    return {
        'model': {
            'hidden_dim': 256,
            'gnn_layers': 3,
            'gnn_type': 'gat',
            'gnn_heads': 4,
            'attention_heads': 8,
            'dropout': 0.2,
            'pooling': 'mean',
            'use_cross_attention': True,
            'use_self_attention': True,
        },
        'data': {
            'dataset': 'davis',
            'protein_model': 'facebook/esm2_t12_35M_UR50D',
            'batch_size': 32,
            'num_workers': 0,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_seed': 42,
        },
        'training': {
            'n_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 20,
            'gradient_clip': 1.0,
            'mixed_precision': False,
        },
        'paths': {
            'data_dir': './data',
            'checkpoint_dir': './checkpoints',
            'log_dir': './runs',
        },
    }
