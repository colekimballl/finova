# dontshare.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Phemex API Credentials
PEMHEPX_API_KEY = os.getenv("PEMHEPX_API_KEY")
PEMHEPX_API_SECRET = os.getenv("PEMHEPX_API_SECRET")

# Ensure that API keys are set
if not PEMHEPX_API_KEY or not PEMHEPX_API_SECRET:
    raise ValueError("Phemex API credentials are not set in environment variables.")

