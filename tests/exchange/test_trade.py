# scripts/common/test_trade.py

import logging
from common.config.config_loader import load_config
from common.logger.logger import setup_logger
from common.interfaces.phemex import PhemexInterface
from common.interfaces.hyperliquid import HyperliquidInterface

def test_trade():
    # Load configuration
    config = load_config(config_path="configs/config.yaml", interactive=False)
    
    # Setup logger
    logger = setup_logger("trade_test", "logs/trade_test.log", level=logging.DEBUG)
    
    # Define symbol and trade parameters
    symbol = "BTC/USD"
    side = "buy"  # or "sell"
    order_type = "market"  # or "limit"
    amount = 0.001  # Example: 0.001 BTC
    price = None  # Not needed for market orders
    
    # Initialize Phemex Interface
    phemex_config = config.get("phemex", {})
    phemex = PhemexInterface(
        api_key=phemex_config.get("api_key"),
        api_secret=phemex_config.get("api_secret")
    )
    
    try:
        phemex.exchange.load_markets()
        # Place a market order
        order_response = phemex.place_order(symbol, side, order_type, amount, price)
        if order_response:
            logger.info(f"Phemex Order Response: {order_response}")
        else:
            logger.error("Failed to place Phemex order.")
    except Exception as e:
        logger.error(f"Error during Phemex trade: {e}")
    
    # Initialize Hyperliquid Interface
    hyperliquid_config = config.get("hyperliquid", {})
    hyperliquid = HyperliquidInterface(
        api_key=hyperliquid_config.get("api_key"),
        api_secret=hyperliquid_config.get("api_secret")
    )
    
    try:
        hyperliquid.exchange.load_markets()
        # Place a market order
        order_response = hyperliquid.place_order(symbol, side, order_type, amount, price)
        if order_response:
            logger.info(f"Hyperliquid Order Response: {order_response}")
        else:
            logger.error("Failed to place Hyperliquid order.")
    except Exception as e:
        logger.error(f"Error during Hyperliquid trade: {e}")

if __name__ == "__main__":
    test_trade()

