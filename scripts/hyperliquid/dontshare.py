# dontshare.py

import os

# It's recommended to use environment variables for sensitive data
PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "default_private_key")
# Add other sensitive configurations as needed
