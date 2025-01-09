# phemex_functions.py

import time
from typing import Tuple, Optional
import pandas as pd
import ccxt
from logger import setup_logger
from exchange_interface_phemex import PhemexClient
import pandas_ta as ta

logger = setup_logger("phemex_functions")

def ask_bid(exchange: PhemexClient, symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetches the current ask and bid prices for a given symbol.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :return: Tuple containing (ask, bid) prices.
    """
    try:
        order_book = exchange.fetch_order_book(symbol)
        if not order_book:
            logger.error(f"Order book for {symbol} is empty.")
            return None, None
        bid = order_book['bids'][0][0] if order_book['bids'] else None
        ask = order_book['asks'][0][0] if order_book['asks'] else None
        logger.debug(f"Ask: {ask}, Bid: {bid} for {symbol}")
        return ask, bid
    except Exception as e:
        logger.error(f"Error fetching ask/bid for {symbol}: {e}")
        return None, None

def open_positions(exchange: PhemexClient, symbol: str) -> Tuple[list, bool, float, bool, int]:
    """
    Retrieves open positions for a specific symbol.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :return: Tuple containing (positions, is_open, size, is_long, index_pos).
    """
    symbol_indices = {
        "uBTCUSD": 4,
        "APEUSD": 2,
        "ETHUSD": 3,
        "DOGEUSD": 1,
        "u100000SHIBUSD": 0,
    }
    index_pos = symbol_indices.get(symbol)
    if index_pos is None:
        logger.error(f"Unsupported symbol: {symbol}")
        return [], False, 0.0, False, -1

    try:
        params = {"type": "swap", "code": "USD"}
        balance = exchange.fetch_balance(params=params)
        positions = balance["info"]["data"]["positions"]
        if index_pos >= len(positions):
            logger.error(f"Index position {index_pos} out of range for symbol {symbol}.")
            return positions, False, 0.0, False, index_pos

        position = positions[index_pos]
        side = position.get("side")
        size = float(position.get("size", 0))
        is_long = side == "Buy"
        is_open = size != 0.0

        logger.debug(f"Open Positions: {positions}, is_open: {is_open}, size: {size}, is_long: {is_long}")
        return positions, is_open, size, is_long, index_pos
    except Exception as e:
        logger.error(f"Error fetching open positions for {symbol}: {e}")
        return [], False, 0.0, False, index_pos

def kill_switch(exchange: PhemexClient, symbol: str, params: dict) -> None:
    """
    Closes all open positions for a given symbol.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :param params: Additional parameters for order placement.
    """
    logger.warning(f"Initiating kill switch for {symbol}.")
    positions, is_open, size, is_long, _ = open_positions(exchange, symbol)

    while is_open:
        logger.info("Executing kill switch: Cancelling all orders and closing position.")
        success = exchange.cancel_all_orders(symbol)
        if not success:
            logger.error(f"Failed to cancel all orders for {symbol}. Retrying in 30 seconds.")
            time.sleep(30)
            continue

        ask, bid = ask_bid(exchange, symbol)
        if ask is None or bid is None:
            logger.error(f"Failed to retrieve ask/bid for {symbol}. Retrying kill switch in 30 seconds.")
            time.sleep(30)
            continue

        try:
            kill_size = int(size)
            if not is_long:
                # Close short position by buying
                exchange.place_order(symbol, "buy", "limit", kill_size, bid, params)
                logger.info(f"Placed BUY order to close short position: {kill_size} {symbol} at {bid}")
            else:
                # Close long position by selling
                exchange.place_order(symbol, "sell", "limit", kill_size, ask, params)
                logger.info(f"Placed SELL order to close long position: {kill_size} {symbol} at {ask}")
        except Exception as e:
            logger.error(f"Error placing kill switch order for {symbol}: {e}")
            time.sleep(30)
            continue

        time.sleep(30)  # Wait to allow order to fill
        positions, is_open, size, is_long, _ = open_positions(exchange, symbol)

    logger.info(f"Kill switch executed successfully for {symbol}.")

def pnl_close(exchange: PhemexClient, symbol: str, target: float, max_loss: float) -> Tuple[bool, bool, float, bool]:
    """
    Monitors PnL and triggers kill switch if target or max loss is reached.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :param target: PnL target percentage.
    :param max_loss: Maximum acceptable loss percentage.
    :return: Tuple indicating (pnl_close_triggered, in_position, size, is_long).
    """
    logger.info(f"Monitoring PnL for {symbol}.")
    positions, is_open, size, is_long, _ = open_positions(exchange, symbol)

    if not is_open:
        logger.info(f"No open position for {symbol} to monitor PnL.")
        return False, False, 0.0, False

    try:
        position = positions[_]
        side = position.get("side")
        contracts = float(position.get("contracts", 0))
        entry_price = float(position.get("entryPrice", 0))
        leverage = float(position.get("leverage", 1))
        current_price = ask_bid(exchange, symbol)[1]

        if current_price is None:
            logger.error(f"Current price for {symbol} is None. Skipping PnL check.")
            return False, is_open, contracts, is_long

        if side.lower() == "long":
            profit = (current_price - entry_price) * leverage
        else:
            profit = (entry_price - current_price) * leverage

        pnl_percentage = (profit / entry_price) * 100
        logger.info(f"PnL for {symbol}: {pnl_percentage:.2f}%")

        pnl_close_triggered = False

        if pnl_percentage >= target:
            logger.info(f"PnL {pnl_percentage:.2f}% exceeds target {target}%. Initiating kill switch.")
            kill_switch(exchange, symbol, {"timeInForce": "PostOnly"})
            pnl_close_triggered = True
        elif pnl_percentage <= max_loss:
            logger.info(f"PnL {pnl_percentage:.2f}% exceeds max loss {max_loss}%. Initiating kill switch.")
            kill_switch(exchange, symbol, {"timeInForce": "PostOnly"})
            pnl_close_triggered = True

        return pnl_close_triggered, is_open, contracts, is_long

    except Exception as e:
        logger.error(f"Error during PnL calculation for {symbol}: {e}")
        return False, is_open, size, is_long

def acct_bal(exchange: PhemexClient, symbol: str) -> float:
    """
    Retrieves the current account balance.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :return: Account balance.
    """
    try:
        balance = exchange.fetch_balance({"type": "swap", "code": "USD"})
        account_value = float(balance["info"]["data"]["accountValue"])
        logger.info(f"Current Account Value: {account_value}")
        return account_value
    except Exception as e:
        logger.error(f"Error fetching account balance: {e}")
        return 0.0

def adjust_leverage_size_signal(exchange: PhemexClient, symbol: str, leverage: float) -> Tuple[float, float]:
    """
    Adjusts leverage and calculates position size based on account balance.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :param leverage: Desired leverage.
    :return: Tuple containing (leverage, calculated_position_size).
    """
    try:
        account_value = acct_bal(exchange, symbol)
        adjusted_value = account_value * 0.95  # Using 95% of account value
        exchange.exchange.set_leverage(leverage, symbol)
        order_book = exchange.fetch_order_book(symbol)
        current_price = order_book['asks'][0][0] if order_book['asks'] else None

        if current_price is None:
            logger.error(f"Current price for {symbol} is None. Cannot calculate position size.")
            return leverage, 0.0

        position_size = (adjusted_value / current_price) * leverage
        position_size = round(float(position_size), 2)  # Adjust decimals as needed
        logger.info(f"Adjusted Leverage: {leverage}x, Calculated Position Size: {position_size}")
        return leverage, position_size
    except Exception as e:
        logger.error(f"Error adjusting leverage and calculating position size: {e}")
        return leverage, 0.0

def df_rsi(exchange: PhemexClient, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Calculates RSI based on historical price data.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :param timeframe: Timeframe for OHLCV data (e.g., '15m', '1d').
    :param limit: Number of data points to fetch.
    :return: DataFrame containing RSI values.
    """
    try:
        bars = exchange.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)
        logger.info(f"Calculated RSI for {symbol} on {timeframe} timeframe.")
        return df
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}: {e}")
        return pd.DataFrame()

def df_sma(exchange: PhemexClient, symbol: str, timeframe: str, limit: int, sma_window: int) -> pd.DataFrame:
    """
    Calculates Simple Moving Average (SMA) based on historical price data.

    :param exchange: Instance of PhemexClient.
    :param symbol: Trading pair.
    :param timeframe: Timeframe for OHLCV data (e.g., '15m', '1d').
    :param limit: Number of data points to fetch.
    :param sma_window: Window size for SMA calculation.
    :return: DataFrame containing SMA values and trading signals.
    """
    try:
        bars = exchange.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[f"sma{sma_window}_{timeframe}"] = df["close"].rolling(window=sma_window).mean()

        ask, bid = ask_bid(exchange, symbol)
        if bid is None:
            logger.error(f"Bid price for {symbol} is None. Cannot generate signals.")
            return df

        # Generate trading signals based on SMA
        df["sig"] = df.apply(
            lambda row: "SELL" if row[f"sma{sma_window}_{timeframe}"] > bid else "BUY",
            axis=1
        )

        # Calculate support and resistance
        df["support"] = df["close"].rolling(window=20).min()
        df["resistance"] = df["close"].rolling(window=20).max()

        logger.info(f"Calculated SMA and trading signals for {symbol} on {timeframe} timeframe.")
        return df
    except Exception as e:
        logger.error(f"Error calculating SMA for {symbol}: {e}")
        return pd.DataFrame()

