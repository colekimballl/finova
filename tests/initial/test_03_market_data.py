import ccxt 
import pytest
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class TestMarketData:
    @pytest.fixture(autouse=True)
    def setup(self, api_credentials):
        """Setup test with logging"""
        logger.info("Setting up market data test...")
        self.symbol = "BTC-USD"
        
    def test_fetch_ticker(self, api_credentials):
        """Test ticker data retrieval"""
        logger.info(f"Fetching ticker for {self.symbol}")
        try:
            exchange = ccxt.coinbaseadvanced({
                'apiKey': api_credentials['api_key'],
                'secret': api_credentials['api_secret'],
                'password': api_credentials['passphrase']
            })
            
            ticker = exchange.fetch_ticker(self.symbol)
            assert ticker is not None
            assert Decimal(str(ticker['last'])) > 0
            
            logger.info(f"Current {self.symbol} price: {ticker['last']}")
        except Exception as e:
            logger.error(f"Failed to fetch ticker: {str(e)}")
            raise
