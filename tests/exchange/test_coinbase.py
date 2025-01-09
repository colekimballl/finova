# tests/exchange/test_coinbase.py

import pytest
import logging
from pathlib import Path
from common.config.config_loader import load_config
from common.interfaces.coinbase import CoinbaseInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCoinbaseIntegration:
    @pytest.fixture(scope="class")
    def client(self):
        """Initialize Coinbase client"""
        try:
            # Get the directory of the current file
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / 'config' / 'test_config.yaml'
            config = load_config(str(config_path))
            if 'coinbase' not in config:
                pytest.skip("Coinbase configuration not found")
            
            logger.info("Initializing Coinbase client with config:")
            logger.info(f"Base URL: {config['coinbase'].get('base_url')}")
            logger.info(f"API Key: {config['coinbase']['api_key'][:10]}...")  # Partial key
            
            return CoinbaseInterface(
                api_key=config['coinbase']['api_key'],
                api_secret=config['coinbase']['api_secret'],
                passphrase=config['coinbase'].get('passphrase', ''),
                base_url=config['coinbase'].get('base_url'),
                testnet=True
            )
        except Exception as e:
            logger.error(f"Error initializing client: {str(e)}")
            pytest.skip(f"Failed to initialize Coinbase client: {str(e)}")

    def test_connectivity(self, client):
        """Test basic connectivity"""
        try:
            ticker = client.exchange.fetch_ticker("BTC/USD")
            assert ticker is not None
            logger.info(f"Successfully connected to Coinbase. BTC Price: {ticker.get('last')}")
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {str(e)}")
            pytest.fail(f"Failed to connect to Coinbase: {str(e)}")

    def test_market_data(self, client):
        """Test market data retrieval"""
        symbol = "BTC/USD"
        try:
            # Test ticker
            ticker = client.exchange.fetch_ticker(symbol)
            assert ticker is not None
            assert 'last' in ticker
            logger.info(f"Current {symbol} price: {ticker['last']}")

            # Test order book
            order_book = client.exchange.fetch_order_book(symbol)
            assert order_book is not None
            assert 'bids' in order_book
            assert 'asks' in order_book
            logger.info(f"Order book depth - Bids: {len(order_book['bids'])}, Asks: {len(order_book['asks'])}")
        except Exception as e:
            logger.error(f"Failed to fetch market data: {str(e)}")
            pytest.fail(f"Failed to fetch market data: {str(e)}")

    def test_account_data(self, client):
        """Test account data access"""
        try:
            balance = client.exchange.fetch_balance()
            assert balance is not None
            logger.info(f"Account balance fetched successfully")
            
            for currency, amount in balance['total'].items():
                if amount and float(amount) > 0:
                    logger.info(f"{currency}: {amount}")
        except Exception as e:
            logger.error(f"Failed to fetch account data: {str(e)}")
            pytest.fail(f"Failed to fetch account data: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

