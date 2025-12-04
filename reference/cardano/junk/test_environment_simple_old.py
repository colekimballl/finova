#!/usr/bin/env python
'''
Simple Environment Test Script
This script checks if all required packages are installed
'''

import sys
import importlib

# Colorize output if possible
try:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
except:
    GREEN = ''
    YELLOW = ''
    RED = ''
    RESET = ''

def print_success(message):
    print(f"{GREEN}✅ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠️ {message}{RESET}")

def print_error(message):
    print(f"{RED}❌ {message}{RESET}")

print("🔍 Testing Python environment for Cardano Trading Bot...")
print(f"Python version: {sys.version}")

# Check required packages
required_packages = [
    "pandas", 
    "numpy", 
    "matplotlib", 
    "requests", 
    "dotenv",
    "pandas_ta"
]

optional_packages = [
    "ccxt"
]

all_required_installed = True

print("\nRequired packages:")
for package in required_packages:
    try:
        # Special case for dotenv which is imported as python-dotenv
        if package == "dotenv":
            importlib.import_module("dotenv")
        else:
            importlib.import_module(package)
        print_success(f"Package {package} is installed")
    except ImportError:
        print_error(f"Package {package} is not installed")
        all_required_installed = False

print("\nOptional packages:")
for package in optional_packages:
    try:
        importlib.import_module(package)
        print_success(f"Package {package} is installed")
    except ImportError:
        print_warning(f"Optional package {package} is not installed")

print("\nData directories:")
import os
dirs_to_check = ["data", "data/cardano", "logs"]
for d in dirs_to_check:
    if os.path.exists(d) and os.path.isdir(d):
        print_success(f"Directory {d} exists")
    else:
        print_warning(f"Directory {d} doesn't exist")
        try:
            os.makedirs(d, exist_ok=True)
            print_success(f"Created directory {d}")
        except:
            print_error(f"Failed to create directory {d}")

# Check .env file
if os.path.exists(".env"):
    print_success(".env file exists")
    
    # Try to load it
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        
        if api_key and api_secret:
            print_success("API credentials found in .env file")
        else:
            print_warning("API credentials not found in .env file")
    except Exception as e:
        print_error(f"Error loading .env file: {str(e)}")
else:
    print_warning(".env file not found. Creating a template...")
    
    try:
        with open(".env", "w") as f:
            f.write("""# Coinbase API credentials
COINBASE_API_KEY="organizations/{org_id}/apiKeys/{key_id}"
COINBASE_API_SECRET="your-secret-key-here"

# Data Collection Settings
HISTORICAL_WEEKS=52  # How many weeks of historical data to fetch initially
""")
        print_success("Created .env template file. Please edit it with your API credentials.")
    except Exception as e:
        print_error(f"Error creating .env template: {str(e)}")

# Final verdict
if all_required_installed:
    print_success("\n🚀 Your environment is ready for the Cardano Trading Bot!")
else:
    print_error("\n⚠️ Some required packages are missing. Please install them before continuing.")
    print("Run the fix_conda_env.sh script to install missing packages.")
