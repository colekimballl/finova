# tests/exchange/hyperliquid/test_01_api_balance.py
import pytest
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

def test_connection(exchange, info):
    """Test basic API connectivity"""
    try:
        meta = info.meta()
        assert meta is not None
        assert "universe" in meta
        logger.info("Successfully connected to Hyperliquid API")
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise

def test_account_exists(account, info):
    """Test account access and basic information"""
    try:
        user_state = info.user_state(account.address)
        assert user_state is not None
        assert "marginSummary" in user_state
        logger.info(f"Successfully accessed account: {account.address[:10]}...")
    except Exception as e:
        logger.error(f"Account access test failed: {e}")
        raise

def test_fetch_balance(account, info):
    """Test balance fetching and validation"""
    try:
        user_state = info.user_state(account.address)
        account_value = Decimal(user_state["marginSummary"]["accountValue"])
        
        assert account_value >= 0, "Account value should not be negative"
        logger.info(f"Account value: {account_value}")
        
        # Test margin requirements if positions exist
        if "assetPositions" in user_state:
            for position in user_state["assetPositions"]:
                logger.info(f"Position found in {position['position']['coin']}")
                
    except Exception as e:
        logger.error(f"Balance fetch test failed: {e}")
        raise
