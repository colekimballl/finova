import ccxt
import pandas as pd
import time
import schedule
from typing import Tuple, Optional


def open_positions(
    phemex: ccxt.phemex, symbol: str
) -> Tuple[list, bool, str, bool, int]:
    """Get position info from Phemex"""
    symbol_indices = {
        "uBTCUSD": 4,
        "APEUSD": 2,
        "ETHUSD": 3,
        "DOGEUSD": 1,
        "u100000SHIBUSD": 0,
    }

    index_pos = symbol_indices.get(symbol)
    if index_pos is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    params = {"type": "swap", "code": "USD"}
    phe_bal = phemex.fetch_balance(params=params)
    open_positions = phe_bal["info"]["data"]["positions"]

    position = open_positions[index_pos]
    openpos_side = position["side"]
    openpos_size = position["size"]

    is_long = openpos_side == "Buy"
    is_open = openpos_size != "0"

    return open_positions, is_open, openpos_size, is_long, index_pos


def ask_bid(phemex: ccxt.phemex, symbol: str) -> Tuple[float, float]:
    """Get current ask/bid prices"""
    ob = phemex.fetch_order_book(symbol)
    bid = ob["bids"][0][0]
    ask = ob["asks"][0][0]
    return ask, bid


def kill_switch(phemex: ccxt.phemex, symbol: str, params: dict) -> None:
    """Close position at market"""
    print(f"starting the kill switch for {symbol}")

    openposi, long, kill_size = open_positions(phemex, symbol)[1:4]
    print(f"openposi {openposi}, long {long}, size {kill_size}")

    while openposi:
        print("starting kill switch loop til limit fil..")

        phemex.cancel_all_orders(symbol)
        openposi, long, kill_size = open_positions(phemex, symbol)[1:4]
        kill_size = int(kill_size)

        ask, bid = ask_bid(phemex, symbol)

        if not long:
            phemex.create_limit_buy_order(symbol, kill_size, bid, params)
            print(f"just made a BUY to CLOSE order of {kill_size} {symbol} at ${bid}")
        else:
            phemex.create_limit_sell_order(symbol, kill_size, ask, params)
            print(f"just made a SELL to CLOSE order of {kill_size} {symbol} at ${ask}")

        print("sleeping for 30 seconds to see if it fills..")
        time.sleep(30)
        openposi = open_positions(phemex, symbol)[1]


def pnl_close(
    phemex: ccxt.phemex, symbol: str, target: float = 9, max_loss: float = -8
) -> Tuple[bool, bool, str, bool]:
    """Monitor PNL and close if target or stop hit"""
    print(f"checking to see if its time to exit for {symbol}... ")

    params = {"type": "swap", "code": "USD"}
    pos_dict = phemex.fetch_positions(params=params)

    index_pos = open_positions(phemex, symbol)[4]
    position = pos_dict[index_pos]

    side = position["side"]
    size = position["contracts"]
    entry_price = float(position["entryPrice"])
    leverage = float(position["leverage"])
    current_price = ask_bid(phemex, symbol)[1]

    print(f"side: {side} | entry_price: {entry_price} | lev: {leverage}")

    if side == "long":
        diff = current_price - entry_price
        long = True
    else:
        diff = entry_price - current_price
        long = False

    try:
        perc = round(((diff / entry_price) * leverage), 10)
    except:
        perc = 0

    perc = 100 * perc
    print(f"for {symbol} this is our PNL percentage: {(perc)}%")

    pnlclose = False
    in_pos = False

    if perc > 0:
        in_pos = True
        print(f"for {symbol} we are in a winning postion")
        if perc > target:
            print(
                ":) :) we are in profit & hit target.. checking volume to see if we should start kill switch"
            )
            pnlclose = True
            kill_switch(phemex, symbol, params)
        else:
            print("we have not hit our target yet")

    elif perc < 0:
        in_pos = True
        if perc <= max_loss:
            print(
                f"we need to exit now down {perc}... so starting the kill switch.. max loss {max_loss}"
            )
            kill_switch(phemex, symbol, params)
        else:
            print(
                f"we are in a losing position of {perc}.. but chillen cause max loss is {max_loss}"
            )
    else:
        print("we are not in position")

    print(f" for {symbol} just finished checking PNL close..")
    return pnlclose, in_pos, size, long


def size_kill(phemex: ccxt.phemex, symbol: str, max_risk: float = 1000) -> None:
    """Emergency kill switch based on position size"""
    params = {"type": "swap", "code": "USD"}
    all_phe_balance = phemex.fetch_balance(params=params)
    open_positions = all_phe_balance["info"]["data"]["positions"]

    try:
        pos_cost = float(open_positions[0]["posCost"])
        openpos_side = open_positions[0]["side"]
        openpos_size = open_positions[0]["size"]
    except:
        pos_cost = 0
        openpos_side = 0
        openpos_size = 0

    print(f"position cost: {pos_cost}")
    print(f"openpos_side : {openpos_side}")

    if pos_cost > max_risk:
        print(
            f"EMERGENCY KILL SWITCH ACTIVATED DUE TO CURRENT POSITION SIZE OF {pos_cost} OVER MAX RISK OF: {max_risk}"
        )
        kill_switch(phemex, symbol, params)
        time.sleep(30000)
    else:
        print(f"size kill check: current position cost is: {pos_cost} we are gucci")
