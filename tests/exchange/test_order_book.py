# scripts/common/test_order_book.py

import logging
from common.config.config_loader import load_config
from common.logger.logger import setup_logger
from common.interfaces.phemex import PhemexInterface
from common.interfaces.hyperliquid import HyperliquidInterface

def test_order_book():
    # Load configuration
    config = load_config(config_path="configs/config.yaml", interactive=False)
    
    # Setup logger
    logger = setup_logger("order_book_test", "logs/order_book_test.log", level=logging.DEBUG)
    
    # Define symbol to test
    symbol = "BTC/USD"
    
    # Initialize Phemex Interface
    phemex_config = config.get("phemex", {})
    phemex = PhemexInterface(
        api_key=phemex_config.get("api_key"),
        api_secret=phemex_config.get("api_secret")
    )
    
    try:
        phemex.exchange.load_markets()
        order_book_phemex = phemex.get_order_book(symbol)
        logger.info(f"Phemex Order Book for {symbol}: {order_book_phemex}")
    except Exception as e:
        logger.error(f"Error fetching Phemex order book: {e}")
    
    # Initialize Hyperliquid Interface
    hyperliquid_config = config.get("hyperliquid", {})
    hyperliquid = HyperliquidInterface(
        api_key=hyperliquid_config.get("api_key"),
        api_secret=hyperliquid_config.get("api_secret")
    )
    
    try:
        hyperliquid.exchange.load_markets()
        order_book_hyperliquid = hyperliquid.get_order_book(symbol)
        logger.info(f"Hyperliquid Order Book for {symbol}: {order_book_hyperliquid}")
    except Exception as e:
        logger.error(f"Error fetching Hyperliquid order book: {e}")

if __name__ == "__main__":
    test_order_book()

