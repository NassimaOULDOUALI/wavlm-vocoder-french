"""
Configuration Utilities
=======================
"""

import yaml
from omegaconf import OmegaConf
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Supports inheritance via _base_ key.
    
    Args:
        config_path: Path to config file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle inheritance
    if '_base_' in config_dict:
        base_path = config_path.parent / config_dict['_base_']
        base_config = load_config(base_path)
        
        # Merge configs (current overrides base)
        config = OmegaConf.merge(base_config, OmegaConf.create(config_dict))
    else:
        config = OmegaConf.create(config_dict)
    
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: OmegaConf configuration object
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)
