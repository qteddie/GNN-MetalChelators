#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration management for MetalGCN using YAML format
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    version: str
    output_size: int = 1
    num_features: int = 66
    hidden_size: int = 384
    dropout: float = 0.0
    depth: int = 4
    heads: int = 8
    max_ligands: int = 3
    metal_emb_dim: int = 8

@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""
    num_epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 5e-5
    anneal_rate: float = 0.9
    huber_beta: float = 0.5
    reg_weight: float = 5.0
    seed: int = 42

@dataclass
class DataConfig:
    """Data paths and preprocessing configuration"""
    metal_csv: str = "../data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv"
    dataloader_path: str = "../data/dataloader.pt"
    data_dir: str = "../data/metal_pka_example"
    proton_model: str = "../ckpt/pka_ver26_best.pkl"

@dataclass
class PathConfig:
    """Output paths configuration"""
    output_dir: str = "../output/metal_ver1"

@dataclass
class MetalGCNConfig:
    """Complete configuration for MetalGCN"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for backward compatibility"""
        result = {}
        result.update(asdict(self.model))
        result.update(asdict(self.training))
        result.update(asdict(self.data))
        result.update(asdict(self.paths))
        return result

def load_config_from_yaml(config_path: str, version: str = None) -> MetalGCNConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        version: Version name to load (if None, loads entire file as config)
        
    Returns:
        MetalGCNConfig: Complete configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # If version is specified, extract that version's config
    if version:
        if version not in data:
            available_versions = list(data.keys())
            raise ValueError(f"Version '{version}' not found in config. Available versions: {available_versions}")
        data = data[version]
    
    return MetalGCNConfig(
        model=ModelConfig(**data.get('model', {})),
        training=TrainingConfig(**data.get('training', {})),
        data=DataConfig(**data.get('data', {})),
        paths=PathConfig(**data.get('paths', {}))
    )

def load_config_by_version(version: str, config_dir: str = "../configs", config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration by version name from unified config file
    
    Args:
        version: Version name (e.g., "metal_ver1")
        config_dir: Directory containing unified config file
        config_file: Name of the unified config file
        
    Returns:
        Dict: Configuration dictionary for backward compatibility
    """
    config_path = os.path.join(config_dir, config_file)
    config = load_config_from_yaml(config_path, version)
    return config.to_dict()

def save_config_to_yaml(config: MetalGCNConfig, config_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the YAML file
    """
    data = {
        'model': asdict(config.model),
        'training': asdict(config.training),
        'data': asdict(config.data),
        'paths': asdict(config.paths)
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def create_config_template(version: str, output_path: str):
    """
    Create a template configuration file
    
    Args:
        version: Version name
        output_path: Path to save the template
    """
    template_config = MetalGCNConfig(
        model=ModelConfig(version=version),
        training=TrainingConfig(),
        data=DataConfig(),
        paths=PathConfig(output_dir=f"../output/{version}")
    )
    
    save_config_to_yaml(template_config, output_path)
    print(f"Configuration template created: {output_path}")

def list_available_versions(config_dir: str = "../configs", config_file: str = "config.yaml") -> List[str]:
    """
    List all available versions in the unified config file
    
    Args:
        config_dir: Directory containing unified config file
        config_file: Name of the unified config file
        
    Returns:
        List of available version names
    """
    config_path = os.path.join(config_dir, config_file)
    if not os.path.exists(config_path):
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return list(data.keys())

# Backward compatibility aliases
load_config_from_toml = load_config_from_yaml
save_config_to_toml = save_config_to_yaml

if __name__ == "__main__":
    # Example usage with unified config
    config_path = "../configs/config.yaml"
    
    # List available versions
    versions = list_available_versions()
    print(f"Available versions: {versions}")
    
    # Test loading specific version
    config = load_config_from_yaml(config_path, "metal_ver1")
    print("\nLoaded configuration (metal_ver1):")
    print(f"Version: {config.model.version}")
    print(f"Hidden size: {config.model.hidden_size}")
    print(f"Learning rate: {config.training.lr}")
    
    # Test loading another version
    config11 = load_config_from_yaml(config_path, "metal_ver11")
    print("\nLoaded configuration (metal_ver11):")
    print(f"Version: {config11.model.version}")
    print(f"Learning rate: {config11.training.lr}")
    print(f"Seed: {config11.training.seed}")