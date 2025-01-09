# tests/exchange/test_connectivity.py
import pytest
import os
from common.config.config_loader import load_config
from common.interfaces.phemex import PhemexInterface
from common.interfaces.hyperliquid import HyperliquidInterface

@pytest.fixture
def config():
    """Test configuration fixture"""
    return load_config("tests/config/test_config.yaml")

def test_phemex_connectivity(config):
    """Test Phemex connection"""
    if not config['phemex']['api_key'] or not config['phemex']['api_secret']:
        pytest.skip("Phemex API credentials not configured")
        
    phemex = PhemexInterface(
        api_key=config['phemex']['api_key'],
        api_secret=config['phemex']['api_secret']
    )
    
    # Test connection
    try:
        markets = phemex.exchange.load_markets()
        assert markets is not None, "Failed to load Phemex markets"
    except Exception as e:
        pytest.fail(f"Failed to connect to Phemex: {str(e)}")

def test_hyperliquid_connectivity(config):
    """Test Hyperliquid connection"""
    if not config['hyperliquid']['api_key'] or not config['hyperliquid']['api_secret']:
        pytest.skip("Hyperliquid API credentials not configured")
        
    hyperliquid = HyperliquidInterface(
        api_key=config['hyperliquid']['api_key'],
        api_secret=config['hyperliquid']['api_secret']
    )
    
    try:
        markets = hyperliquid.exchange.load_markets()
        assert markets is not None, "Failed to load Hyperliquid markets"
    except Exception as e:
        pytest.fail(f"Failed to connect to Hyperliquid: {str(e)}")
