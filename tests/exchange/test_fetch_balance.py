# scripts/common/test_fetch_balance.py

import logging
from common.config.config_loader import load_config
from common.logger.logger import setup_logger
from common.interfaces.phemex import PhemexInterface
from common.interfaces.hyperliquid import HyperliquidInterface

def test_fetch_balance():
    # Load configuration
    config = load_config(config_path="configs/config.yaml", interactive=False)
    
    # Setup logger
    logger = setup_logger("balance_test", "logs/balance_test.log", level=logging.DEBUG)
    
    # Initialize Phemex Interface
    phemex_config = config.get("phemex", {})
    phemex = PhemexInterface(
        api_key=phemex_config.get("api_key"),
        api_secret=phemex_config.get("api_secret")
    )
    
    try:
        phemex.exchange.load_markets()
        balance_phemex = phemex.get_balance()
        logger.info(f"Phemex Balance: {balance_phemex}")
    except Exception as e:
        logger.error(f"Error fetching Phemex balance: {e}")
    
    # Initialize Hyperliquid Interface
    hyperliquid_config = config.get("hyperliquid", {})
    hyperliquid = HyperliquidInterface(
        api_key=hyperliquid_config.get("api_key"),
        api_secret=hyperliquid_config.get("api_secret")
    )
    
    try:
        hyperliquid.exchange.load_markets()
        balance_hyperliquid = hyperliquid.get_balance()
        logger.info(f"Hyperliquid Balance: {balance_hyperliquid}")
    except Exception as e:
        logger.error(f"Error fetching Hyperliquid balance: {e}")

if __name__ == "__main__":
    test_fetch_balance()

