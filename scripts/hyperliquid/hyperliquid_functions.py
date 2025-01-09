# hyperliquid_functions.py

import json
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import pandas_ta as ta
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import schedule

import dontshare as d  # Ensure this contains only necessary configurations

def ask_bid(symbol: str):
    """Fetches the ask and bid for a given symbol."""
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {"type": "l2Book", "coin": symbol}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        l2_data = response.json().get("levels", [])
        if len(l2_data) >= 2:
            bid = float(l2_data[0][0]["px"])
            ask = float(l2_data[1][0]["px"])
            return ask, bid, l2_data
        else:
            print("Insufficient level 2 data received.")
            return None, None, l2_data
    else:
        print(f"Error fetching ask/bid for {symbol}: {response.status_code}")
        return None, None, []

def get_sz_px_decimals(symbol: str):
    """Returns size decimals and price decimals for a given symbol."""
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    data = {"type": "meta"}

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        data = response.json()
        symbols = data.get("universe", [])
        symbol_info = next((s for s in symbols if s["name"] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info.get("szDecimals", 0)
        else:
            print("Symbol not found in metadata.")
            sz_decimals = 0
    else:
        print(f"Error fetching metadata: {response.status_code}")
        sz_decimals = 0

    ask = ask_bid(symbol)[0]
    if ask is not None:
        px_decimals = len(str(ask).split(".")[1]) if "." in str(ask) else 0
    else:
        px_decimals = 0

    print(f"{symbol} - Size Decimals: {sz_decimals}, Price Decimals: {px_decimals}")
    return sz_decimals, px_decimals

def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool, account: LocalAccount):
    """Places a limit order."""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    sz_decimals = get_sz_px_decimals(coin)[0]
    sz = round(sz, sz_decimals)

    print(f"Placing {'BUY' if is_buy else 'SELL'} order for {coin}: Size={sz}, Price={limit_px}")
    order_result = exchange.order(
        coin, is_buy, sz, limit_px, {"limit": {"tif": "GTC"}}, reduce_only=reduce_only
    )

    order_type = "BUY" if is_buy else "SELL"
    status = order_result['response']['data']['statuses'][0] if 'response' in order_result else "Unknown"
    print(f"Limit {order_type} order placed. Status: {status}")

    return order_result

def cancel_all_orders(account: LocalAccount):
    """Cancels all open orders for the account."""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    open_orders = info.open_orders(account.address)

    if not open_orders:
        print("No open orders to cancel.")
        return

    print("Cancelling all open orders...")
    for open_order in open_orders:
        exchange.cancel(open_order["coin"], open_order["oid"])
        print(f"Cancelled order {open_order['oid']} for {open_order['coin']}")

def get_position(symbol: str, account: LocalAccount):
    """Retrieves current position information for a given symbol."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    print(f"Account Value: {user_state['marginSummary']['accountValue']}")
    positions = []
    in_pos = False
    size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long = None

    for position in user_state.get("assetPositions", []):
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f"Position PnL: {pnl_perc}%")
            break

    if size > 0:
        long = True
    elif size < 0:
        long = False

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def kill_switch(symbol: str, account: LocalAccount):
    """Closes all positions for a given symbol."""
    positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    while in_pos:
        cancel_all_orders(account)
        ask, bid, l2 = ask_bid(symbol)
        pos_size = abs(pos_size)

        if long:
            limit_order(pos_sym, False, pos_size, ask, True, account)
            print("Kill switch: SELL to close position.")
        elif not long:
            limit_order(pos_sym, True, pos_size, bid, True, account)
            print("Kill switch: BUY to close position.")

        time.sleep(5)
        positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print("Position successfully closed via kill switch.")

def pnl_close(symbol: str, target: float, max_loss: float, account: LocalAccount):
    """Monitors PnL and closes position if target or max loss is reached."""
    print("Monitoring PnL for position closure.")
    positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    if pnl_perc > target:
        print(f"PnL {pnl_perc}% exceeds target {target}%. Closing position as WIN.")
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f"PnL {pnl_perc}% exceeds max loss {max_loss}%. Closing position as LOSS.")
        kill_switch(pos_sym, account)
    else:
        print(f"PnL {pnl_perc}% does not meet closure criteria.")

def acct_bal(account: LocalAccount) -> float:
    """Retrieves the current account balance."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    acct_value = float(user_state["marginSummary"]["accountValue"])
    print(f"Current Account Value: {acct_value}")
    return acct_value

def adjust_leverage_size_signal(symbol: str, leverage: float, account: LocalAccount):
    """Calculates position size based on account balance and leverage."""
    print(f"Adjusting leverage to {leverage}x for {symbol}.")
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    acct_value = float(user_state["marginSummary"]["accountValue"])
    acct_val95 = acct_value * 0.95

    exchange.update_leverage(leverage, symbol)
    price = ask_bid(symbol)[0]

    size = (acct_val95 / price) * leverage
    sz_decimals = get_sz_px_decimals(symbol)[0]
    size = round(float(size), sz_decimals)

    print(f"Calculated position size: {size}")
    return leverage, size

