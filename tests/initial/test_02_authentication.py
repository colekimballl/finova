# test_02_authentication.py
import pytest
import logging
from common.interfaces.coinbase import CoinbaseInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_coinbase.log')
    ]
)
logger = logging.getLogger(__name__)

class TestCoinbaseAuth:
    def test_api_key_auth(self, api_credentials):
        """Test API key authentication"""
        logger.info("Testing API key authentication...")
        client = CoinbaseInterface(
            api_key=api_credentials['api_key'],
            api_secret=api_credentials['api_secret'],
            passphrase=api_credentials['passphrase']
        )
        
        try:
            # Try to access a private endpoint
            accounts = client.get_accounts()
            assert accounts is not None
            logger.info("✅ API key authentication successful")
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            raise

    def test_invalid_auth(self):
        """Test behavior with invalid credentials"""
        logger.info("Testing invalid authentication...")
        client = CoinbaseInterface(
            api_key="invalid_key",
            api_secret="invalid_secret",
            passphrase="invalid_passphrase"
        )
        
        with pytest.raises(Exception) as exc_info:
            client.get_accounts()
        logger.info("✅ Invalid authentication properly handled")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
