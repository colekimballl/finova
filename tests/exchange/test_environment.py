# tests/exchange/test_environment.py

import pytest
import sys
import os
import yaml
import logging
from pathlib import Path

def setup_test_environment():
    """Configure testing environment with proper paths and configurations."""
    project_root = Path(__file__).parent.parent.parent  # Adjust to get correct root
    sys.path.append(str(project_root))
    
    test_config = {
        'phemex': {
            'api_key': os.getenv('PHEMEX_API_KEY', 'test_key'),
            'api_secret': os.getenv('PHEMEX_API_SECRET', 'test_secret'),
            'base_url': 'https://testnet-api.phemex.com'
        },
        'hyperliquid': {
            'api_key': os.getenv('HYPERLIQUID_API_KEY', 'test_key'),
            'api_secret': os.getenv('HYPERLIQUID_API_SECRET', 'test_secret'),
            'base_url': 'https://testnet-api.hyperliquid.com'
        }
    }
    
    return project_root, test_config

def test_environment_setup():
    """Test environment setup"""
    root, config = setup_test_environment()
    assert root.exists(), "Project root directory not found"
    assert root.joinpath('tests').exists(), "Tests directory not found"
    assert root.joinpath('common').exists(), "Common directory not found"
    
def test_config_structure():
    """Test configuration structure"""
    _, config = setup_test_environment()
    
    # Test Phemex config
    assert 'phemex' in config, "Phemex configuration missing"
    assert 'api_key' in config['phemex'], "Phemex API key missing"
    assert 'api_secret' in config['phemex'], "Phemex API secret missing"
    assert 'base_url' in config['phemex'], "Phemex base URL missing"
    
    # Test Hyperliquid config
    assert 'hyperliquid' in config, "Hyperliquid configuration missing"
    assert 'api_key' in config['hyperliquid'], "Hyperliquid API key missing"
    assert 'api_secret' in config['hyperliquid'], "Hyperliquid API secret missing"
    assert 'base_url' in config['hyperliquid'], "Hyperliquid base URL missing"

def test_logging_setup():
    """Test logging configuration"""
    root, _ = setup_test_environment()
    log_dir = root / 'tests' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = log_dir / 'test.log'
    logging.basicConfig(
        filename=str(log_file),
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    assert log_dir.exists(), "Log directory not created"
    
    # Test logging
    test_logger = logging.getLogger("test")
    test_logger.info("Test log message")
    assert log_file.exists(), "Log file not created"
