# bot.py

import time
import schedule
import pandas as pd
from logger import setup_logger
from exchange_interface_phemex import PhemexClient
from phemex_functions import (
    ask_bid,
    open_positions,
    pnl_close,
    df_sma,
    df_rsi,
    adjust_leverage_size_signal,
)
from risk_management_phemex import RiskParameters, RiskManager
import dontshare

logger = setup_logger("bot")

# Configuration Parameters
SYMBOL = "uBTCUSD"
TIMEFRAME_SMA = "1d"
TIMEFRAME_RSI = "15m"
LIMIT = 100
SMA_WINDOW = 20
RSI_PERIOD = 14
TARGET_PNL = 8.0  # in percentage
MAX_LOSS_PNL = -8.0  # in percentage
LEVERAGE = 3.0
POS_SIZE = 30.0  # Adjust as per strategy
VOL_DECIMAL_THRESHOLD = 0.4  # For volume-based decisions

# Initialize Exchange Client
#fix this

def bot():
    """
    Main bot function executing trading strategy.
    """
    logger.info("Starting bot run.")

    # Connect to Phemex
    if not exchange.connect():
        logger.error("Failed to connect to Phemex. Skipping this bot run.")
        return

    # Initialize Risk Manager
    risk_params = RiskParameters(
        max_position_size=1000.0,  # Example value
        max_drawdown=MAX_LOSS_PNL,
        leverage=LEVERAGE
    )
    risk_manager = RiskManager(risk_params, exchange, SYMBOL)

    # Adjust Leverage and Calculate Position Size
    _, calculated_size = adjust_leverage_size_signal(exchange, SYMBOL, LEVERAGE)

    # Check Current Positions
    positions, is_open, pos_size, is_long, _ = open_positions(exchange, SYMBOL)
    logger.info(f"Current Positions: {positions}")

    # Monitor PnL and Manage Risk
    risk_manager.monitor_pnl()

    # Fetch Current Ask and Bid
    ask, bid = ask_bid(exchange, SYMBOL)
    if ask is None or bid is None:
        logger.error("Failed to retrieve ask/bid prices. Skipping order placement.")
        exchange.disconnect()
        return

    logger.info(f"Current Ask: {ask}, Bid: {bid}")

    # Calculate SMA and Generate Signals
    df_sma_data = df_sma(exchange, SYMBOL, TIMEFRAME_SMA, LIMIT, SMA_WINDOW)
    if df_sma_data.empty:
        logger.error("SMA DataFrame is empty. Skipping this bot run.")
        exchange.disconnect()
        return

    latest_signal = df_sma_data.iloc[-1]["sig"]
    logger.info(f"Latest Trading Signal: {latest_signal}")

    # Trading Logic: Place Orders if Not in Position
    if not is_open and calculated_size < POS_SIZE:
        logger.info("Not in position. Preparing to place new orders based on trading signal.")

        if latest_signal == "BUY":
            # Place Buy Orders
            try:
                logger.info("Placing BUY limit orders.")
                exchange.place_order(SYMBOL, "buy", "limit", calculated_size, bid, {"timeInForce": "PostOnly"})
                exchange.place_order(SYMBOL, "buy", "limit", calculated_size, bid * 0.99, {"timeInForce": "PostOnly"})
                logger.info("BUY orders placed successfully.")
            except Exception as e:
                logger.error(f"Error placing BUY orders: {e}")

        elif latest_signal == "SELL":
            # Place Sell Orders
            try:
                logger.info("Placing SELL limit orders.")
                exchange.place_order(SYMBOL, "sell", "limit", calculated_size, ask, {"timeInForce": "PostOnly"})
                exchange.place_order(SYMBOL, "sell", "limit", calculated_size, ask * 1.01, {"timeInForce": "PostOnly"})
                logger.info("SELL orders placed successfully.")
            except Exception as e:
                logger.error(f"Error placing SELL orders: {e}")

    else:
        logger.info("Already in position or position size exceeds limit. Skipping order placement.")

    # Disconnect from Phemex
    exchange.disconnect()
    logger.info("Bot run completed.")

# Schedule the bot to run every 30 seconds
schedule.every(30).seconds.do(bot)

if __name__ == "__main__":
    logger.info("Starting Phemex Trading Bot.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unhandled exception in main loop: {e}")
            time.sleep(30)

