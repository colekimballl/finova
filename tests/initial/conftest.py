import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

def pytest_sessionstart(session):
    """Load environment variables before tests start"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)

@pytest.fixture(scope="session")
def api_credentials():
    """Provide API credentials to tests"""
    required_vars = ['COINBASE_API_KEY', 'COINBASE_API_SECRET', 'COINBASE_PASSPHRASE']
    
    # Check for missing environment variables
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")
    
    return {
        'api_key': os.getenv('COINBASE_API_KEY'),
        'api_secret': os.getenv('COINBASE_API_SECRET'),
        'passphrase': os.getenv('COINBASE_PASSPHRASE')
    }
