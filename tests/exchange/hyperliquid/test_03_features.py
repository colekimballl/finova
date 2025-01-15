# test_03_features.py
import pytest
import logging
from termcolor import cprint
import json
import time

logger = logging.getLogger(__name__)

def test_order_validation(exchange, info, account):
    """Test order validation without placing actual orders"""
    symbol = "BTC"
    try:
        # Get market info first
        meta = info.meta()
        market = next((m for m in meta["universe"] if m["name"] == symbol), None)
        
        if market:
            cprint("\n🔍 Order Validation Check", "cyan", attrs=["bold"])
            print(f"Market found: {symbol}")
            print(f"Size decimals: {market.get('szDecimals')}")
            print("✓ Market validation successful")
            
    except Exception as e:
        logger.error(f"Order validation test failed: {e}")
        raise

def test_leverage_limits(exchange, info):
    """Test leverage limits without setting leverage"""
    symbol = "BTC"
    try:
        meta = info.meta()
        universe = meta.get("universe", [])
        market_info = next((m for m in universe if m["name"] == symbol), None)
        
        cprint("\n⚙️ Leverage Limits", "cyan", attrs=["bold"])
        if market_info:
            max_leverage = market_info.get("maxLeverage", "N/A")
            print(f"Symbol: {symbol}")
            print(f"Max Leverage: {max_leverage}")
            print("✓ Successfully retrieved leverage limits")
            
            # Don't actually set leverage, just verify we can get the info
            assert max_leverage is not None, "Max leverage information not available"
        else:
            print(f"No market information found for {symbol}")
            
    except Exception as e:
        logger.error(f"Leverage limits test failed: {e}")
        raise

def print_test_summary(passed, failed):
    """Print colorful test summary"""
    cprint("\n============================", "blue", attrs=["bold"])
    cprint("🔍 Hyperliquid Test Summary", "blue", attrs=["bold"])
    cprint("============================", "blue", attrs=["bold"])
    
    cprint(f"\n✅ Passed Tests: {passed}", "green")
    if failed == 0:
        cprint("❌ Failed Tests: 0", "green")
        cprint("\n🎉 All tests completed successfully!", "green", attrs=["bold"])
    else:
        cprint(f"❌ Failed Tests: {failed}", "red")
        cprint("\n⚠️  Some tests failed. Please check the logs.", "yellow", attrs=["bold"])
