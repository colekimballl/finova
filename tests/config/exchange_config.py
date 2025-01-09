# tests/config/exchange_config.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class Exchange(Enum):
    COINBASE = "coinbase"
    PHEMEX = "phemex"
    HYPERLIQUID = "hyperliquid"

@dataclass
class ExchangeConfig:
    name: Exchange
    api_key: str
    api_secret: str
    base_url: str
    is_testnet: bool = False
    extra_params: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_default_urls(cls, exchange: Exchange, testnet: bool = False) -> str:
        urls = {
            Exchange.COINBASE: {
                "main": "https://api.coinbase.com",
                "test": "https://api-public.sandbox.pro.coinbase.com"
            },
            Exchange.PHEMEX: {
                "main": "https://api.phemex.com",
                "test": "https://testnet-api.phemex.com"
            },
            Exchange.HYPERLIQUID: {
                "main": "https://api.hyperliquid.com",
                "test": "https://testnet-api.hyperliquid.com"
            }
        }
        return urls[exchange]["test" if testnet else "main"]

def load_exchange_config(exchange: Exchange, config_path: str = "config.yaml") -> ExchangeConfig:
    """Load exchange specific configuration"""
    import yaml
    import os
    from pathlib import Path
    
    # Load environment variables first
    env_prefix = exchange.value.upper()
    api_key = os.getenv(f"{env_prefix}_API_KEY")
    api_secret = os.getenv(f"{env_prefix}_API_SECRET")
    
    # If environment variables not set, try config file
    if not api_key or not api_secret:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                exchange_config = config.get(exchange.value, {})
                api_key = exchange_config.get("api_key", "")
                api_secret = exchange_config.get("api_secret", "")
    
    return ExchangeConfig(
        name=exchange,
        api_key=api_key,
        api_secret=api_secret,
        base_url=ExchangeConfig.get_default_urls(exchange, testnet=True),
        is_testnet=True
    )
