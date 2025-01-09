# build kill switch and pnl close for hyperliquid

import nice_funcs as n
import key_file as d
from eth_account.signers.local import LocalAccount
import eth_account
import json
import time
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import pandas as pd
import datetime
import schedule
import requests

symbol = "WIF"  # example
max_loss = -5
target = 4
acct_min = 9
timeframe = "4h"
size = 10
coin = symbol
secret_key = d.secret_key
account = LocalAccount - eth_account.Account.from_key(secret_key)


def acct_bal(account):

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    print(
        f'this is current account value: {user_state["marginSummary"]["accountValue"]}'
    )

    acct_value = user_state["marginSummary"]["accountValue"]

    return acct_value


acct_min = 7


def bot():

    print("this is our bot and it is alive")

    print("controlling risk with our pnl close")

    n.pnl_close(symbol, target, max_loss, account)

    # if we have over X positions
    # if account size goes under X like $100 never go under $70

    acct_val = float(acct_bal(account))

    if acct_val < acct_min:
        print(f"account value is {acct_val} and closing because out low is {acct_min}")
        n.kill_switch(symbol, account)


bot()
