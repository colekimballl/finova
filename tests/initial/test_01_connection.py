import pytest
import logging
import ccxt
from common.interfaces.coinbase import CoinbaseInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_connection(api_credentials):
    """Test basic connection to Coinbase API"""
    logger.info("Starting basic connection test...")
    
    # Create exchange directly first to test connection
    try:
        exchange = ccxt.coinbaseadvanced({
            'apiKey': api_credentials['api_key'],
            'secret': api_credentials['api_secret'],
            'password': api_credentials['passphrase'],
            'urls': {
                'api': {
                    'public': 'https://api.coinbase.com',
                    'private': 'https://api.coinbase.com',
                    'rest': 'https://api.coinbase.com'
                }
            }
        })
        
        # Basic public endpoint test
        time = exchange.fetch_time()
        assert time > 0
        logger.info(f"✅ Connected successfully! Server time: {time}")
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        raise
