"""
Utility helpers for configuration management and shared routines.
"""

import argparse
import os
import yaml
import json
from typing import Dict, Any

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load a config file in YAML or JSON format."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments into config values, with CLI taking priority."""
    merged_config = config.copy()
    
    # Override config values with any explicitly provided CLI arguments.
    for key, value in vars(args).items():
        if value is not None and key not in ['command', 'config_file']:
            merged_config[key] = value
    
    return merged_config

class ConfigManager:
    """Configuration manager that centralizes argument handling."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = {}
        
        # Load the config file when one is provided.
        if hasattr(args, 'config_file') and args.config_file:
            self.config = load_config_file(args.config_file)
            print(f"Loaded config file: {args.config_file}")
        
        # Apply CLI overrides.
        self.config = merge_config_with_args(self.config, args)
    
    def get(self, key: str, default=None):
        """Return a single config value."""
        return self.config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Return a copy of the full config."""
        return self.config.copy()
