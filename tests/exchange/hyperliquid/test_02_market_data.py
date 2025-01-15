# test_02_market_data.py
import pytest
import logging
from decimal import Decimal
from termcolor import cprint
import json
import requests

logger = logging.getLogger(__name__)

def test_fetch_orderbook(info):
    """Test L2 order book data retrieval"""
    symbols = ["BTC", "ETH"]
    
    for symbol in symbols:
        try:
            # Direct API call without dependencies
            url = "https://api.hyperliquid.xyz/info"
            headers = {"Content-Type": "application/json"}
            data = {"type": "l2Book", "coin": symbol}
            
            response = requests.post(url, headers=headers, json=data)
            assert response.status_code == 200
            l2_data = response.json().get("levels", [])
            
            cprint(f"\n📊 Order Book - {symbol}", "cyan", attrs=["bold"])
            if len(l2_data) >= 2:
                bid = float(l2_data[0][0]["px"])
                ask = float(l2_data[1][0]["px"])
                cprint("Top Ask:", "red")
                print(f"Price: {ask}")
                cprint("Top Bid:", "green")
                print(f"Price: {bid}")
            else:
                print("Insufficient order book data")
                
        except Exception as e:
            logger.error(f"Order book test failed for {symbol}: {e}")
            raise

def test_check_empty_positions(info, account):
    """Verify position checking for empty account"""
    try:
        user_state = info.user_state(account.address)
        positions = user_state.get("assetPositions", [])
        
        cprint("\n📈 Position Check", "cyan", attrs=["bold"])
        if not positions:
            cprint("✓ Account has no open positions (expected for new account)", "green")
        else:
            cprint("Account has existing positions:", "yellow")
            for pos in positions:
                print(f"Symbol: {pos['position']['coin']}")
                print(f"Size: {pos['position']['szi']}")
        
    except Exception as e:
        logger.error(f"Position check failed: {e}")
        raise

def test_check_empty_orders(info, account):
    """Verify order checking for empty account"""
    try:
        user_state = info.user_state(account.address)
        orders = user_state.get("orders", [])
        
        cprint("\n📋 Open Orders Check", "cyan", attrs=["bold"])
        if not orders:
            cprint("✓ Account has no open orders (expected for new account)", "green")
        else:
            cprint("Account has existing orders:", "yellow")
            for order in orders:
                print(f"Symbol: {order.get('coin')}")
                print(f"Size: {order.get('sz')}")
        
    except Exception as e:
        logger.error(f"Order check failed: {e}")
        raise

def test_market_capabilities(info):
    """Test available trading capabilities without placing orders"""
    try:
        meta = info.meta()
        assert meta is not None
        assert "universe" in meta
        
        cprint("\n🔧 Trading Capabilities", "cyan", attrs=["bold"])
        for market in meta["universe"]:
            symbol = market["name"]
            cprint(f"\nSymbol: {symbol}", "yellow")
            print(f"Size Decimals: {market.get('szDecimals', 'N/A')}")
            print(f"Base Currency: {market.get('baseCurrency', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Market capabilities test failed: {e}")
        raise