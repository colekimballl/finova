# common/bot_template.py

"""
Module Name: <Your New Bot>
Description: Short description of what this bot does.

Author: Your Name
Date: YYYY-MM-DD
"""

import logging
from common.config.config_loader import load_config
from common.logger.logger import setup_logger
from common.api_client import ExchangeClient
from common.risk_management.risk_manager import RiskParameters, RiskManager
from common.indicators.technical_indicators import IndicatorManager
from common.interfaces.phemex import PhemexInterface  # Import specific interface


def main():
    # Load Configuration
    config = load_config()
    exchange_config = config["phemex"]

    # Setup Logger
    logger = setup_logger(
        "bot_template",
        f"logs/phemex/bot_template.log",
        level=logging.INFO
    )

    # Initialize Exchange Interface
    phemex_interface = PhemexInterface(
        api_key=exchange_config["api_key"],
        api_secret=exchange_config["api_secret"]
    )

    if not phemex_interface.connect():
        logger.error("Failed to connect to Phemex. Exiting bot.")
        return

    # Initialize Indicator Manager
    indicator_manager = IndicatorManager()
    indicator_manager.add_indicator(TA_LibSMA(period=20))
    indicator_manager.add_indicator(TA_LibEMA(period=20))
    indicator_manager.add_indicator(PandasTA_RSI(length=14))
    # Add more indicators as needed

    # Initialize Risk Manager
    risk_params = RiskParameters(
        max_position_size=10000.0,  # Example value in USD
        max_drawdown=15.0,           # Example value in percentage
        leverage=3.0                 # Example leverage
    )
    risk_manager = RiskManager(
        exchange=phemex_interface,
        params=risk_params
    )

    # Your bot logic here
    # Example: Fetch data, calculate indicators, make trading decisions

    # Disconnect from Exchange
    phemex_interface.disconnect()
    logger.info("Bot execution completed.")


if __name__ == "__main__":
    main()

