# common/config/config_loader.py

import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

def load_config(config_path: str) -> dict:
    """Load YAML configuration and environment variables from the given path."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    # Load environment variables from .env
    load_dotenv()
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with environment variables if present
    if 'coinbase' in config:
        config['coinbase']['api_key'] = os.getenv('COINBASE_API_KEY', config['coinbase'].get('api_key'))
        config['coinbase']['api_secret'] = os.getenv('COINBASE_API_SECRET', config['coinbase'].get('api_secret'))
        config['coinbase']['passphrase'] = os.getenv('COINBASE_PASSPHRASE', config['coinbase'].get('passphrase'))
    
    return config

