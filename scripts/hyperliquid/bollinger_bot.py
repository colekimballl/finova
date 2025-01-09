# bollinger_bot.py

import time
import schedule
import pandas as pd
import pandas_ta as ta
from eth_account.signers.local import LocalAccount
import eth_account

import hyperliquid_functions as n

# Configuration
symbol = "WIF"
timeframe = "15m"
sma_window = 20
lookback_days = 1
size = 1
target = 5
max_loss = -10
leverage = 3
max_positions = 1

secret = "YOUR_PRIVATE_KEY_HERE"  # Replace with secure key management

def bot():
    # Initialize account
    account = eth_account.Account.from_key(secret)

    # Get current positions and maximum positions
    positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = n.get_position_andmaxpos(
        symbol, account, max_positions
    )
    print(f"Current positions for {symbol}: {positions}")

    # Adjust leverage and calculate position size
    lev, pos_size = n.adjust_leverage_size_signal(symbol, leverage, account)
    pos_size /= 2  # Dividing position by 2

    if in_pos:
        n.cancel_all_orders(account)
        print("In position. Checking PnL to potentially close.")
        n.pnl_close(symbol, target, max_loss, account)
    else:
        print("Not in position. No PnL close needed.")

    # Fetch current ask and bid
    ask, bid, l2_data = n.ask_bid(symbol)
    if ask is None or bid is None:
        print("Failed to retrieve ask/bid. Skipping this iteration.")
        return

    print(f"Ask: {ask}, Bid: {bid}")

    # Fetch and process OHLCV data
    snapshot_data = n.get_ohlcv2("BTC", "1m", 500)
    df = n.process_data_to_df(snapshot_data)
    if df.empty:
        print("No OHLCV data received. Skipping Bollinger Bands calculation.")
        return

    _, bollinger_bands_tight, _ = n.calculate_bollinger_bands(df)
    print(f"Bollinger Bands Tight: {bollinger_bands_tight}")

    # Trading Logic
    if not in_pos and bollinger_bands_tight:
        print("Bollinger Bands are tight and no existing position. Entering new position.")
        n.cancel_all_orders(account)
        print("All open orders canceled.")

        # Place BUY and SELL limit orders
        bid_price = float(l2_data[0][10]["px"]) if len(l2_data[0]) > 10 else bid
        ask_price = float(l2_data[1][10]["px"]) if len(l2_data[1]) > 10 else ask

        n.limit_order(symbol, True, pos_size, bid_price, False, account)
        print(f"Placed BUY order for {pos_size} at {bid_price}")

        n.limit_order(symbol, False, pos_size, ask_price, False, account)
        print(f"Placed SELL order for {pos_size} at {ask_price}")

    elif not bollinger_bands_tight:
        n.cancel_all_orders(account)
        n.close_all_positions(account)
    else:
        print(f"Current position status: {in_pos}. Bollinger Bands may not be tight.")

# Schedule the bot to run every 30 seconds
schedule.every(30).seconds.do(bot)

if __name__ == "__main__":
    bot()  # Initial run
    while True:
        try:
            schedule.run_pending()
            time.sleep(10)
        except Exception as e:
            print("*** Possible internet connection issue. Sleeping for 30 seconds before retrying.")
            print(e)
            time.sleep(30)

