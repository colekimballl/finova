#!/usr/bin/env python
'''
Project Solaris AI - Consolidated Environment Test
-------------------------------------------------
This script tests if your environment is correctly set up by:
1. Checking if all required packages are installed
2. Verifying that the .env file can be loaded
3. Testing API connectivity
4. Testing data access
'''

import os
import sys
import importlib
import platform
from datetime import datetime
import subprocess
import json

# Define colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(message):
    print(f"{GREEN}✅ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠️ {message}{RESET}")

def print_error(message):
    print(f"{RED}❌ {message}{RESET}")

def print_info(message):
    print(f"{BLUE}ℹ️ {message}{RESET}")

def print_header(title):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")

def check_python_version():
    """Check if Python version is compatible"""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    if version_info.major == 3 and version_info.minor >= 9:
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version {version_str} is not compatible. Please use Python 3.9 or higher.")
        return False

def check_conda_environment():
    """Check if running in the cardano conda environment"""
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    
    if env_name == 'cardano':
        print_success(f"Running in conda environment: {env_name}")
        return True
    elif env_name:
        print_warning(f"Running in conda environment '{env_name}', but expected 'cardano'")
        return False
    else:
        print_warning("Not running in a conda environment")
        return False

def check_packages():
    """Check if all required packages are installed"""
    required_packages = [
        {"name": "pandas", "module": "pandas"},
        {"name": "numpy", "module": "numpy"},
        {"name": "matplotlib", "module": "matplotlib"},
        {"name": "requests", "module": "requests"},
        {"name": "python-dotenv", "module": "dotenv"},
        {"name": "pandas-ta", "module": "pandas_ta"}
    ]
    
    optional_packages = [
        {"name": "ccxt", "module": "ccxt"}
    ]
    
    all_required_installed = True
    
    print_info("Checking required packages...")
    for package in required_packages:
        try:
            importlib.import_module(package["module"])
            print_success(f"Package {package['name']} is installed")
        except ImportError:
            print_error(f"Package {package['name']} is not installed")
            all_required_installed = False
    
    print_info("\nChecking optional packages...")
    for package in optional_packages:
        try:
            importlib.import_module(package["module"])
            print_success(f"Package {package['name']} is installed")
        except ImportError:
            print_warning(f"Optional package {package['name']} is not installed")
    
    return all_required_installed

def check_package_versions():
    """Check versions of key packages"""
    packages_to_check = [
        {"name": "pandas", "module": "pandas"},
        {"name": "numpy", "module": "numpy"},
        {"name": "matplotlib", "module": "matplotlib"},
        {"name": "pandas-ta", "module": "pandas_ta"}
    ]
    
    print_info("\nPackage versions:")
    for package in packages_to_check:
        try:
            module = importlib.import_module(package["module"])
            version = getattr(module, "__version__", "unknown")
            print(f"  {package['name']}: {version}")
        except ImportError:
            print(f"  {package['name']}: not installed")

def check_env_file():
    """Check if .env file exists and can be loaded"""
    try:
        from dotenv import load_dotenv
        
        if os.path.exists('.env'):
            load_dotenv()
            
            api_key = os.getenv('COINBASE_API_KEY')
            api_secret = os.getenv('COINBASE_API_SECRET')
            
            if api_key and api_secret:
                print_success(".env file exists and contains API credentials")
                return True
            else:
                print_warning(".env file exists but doesn't contain API credentials")
                return False
        else:
            print_error(".env file not found. Please create one with your API credentials.")
            print_info("Run './setup.sh' to create a template .env file.")
            return False
    except ImportError:
        print_error("python-dotenv package is not installed. Cannot check .env file.")
        return False

def test_coinbase_connectivity():
    """Test connection to Coinbase API"""
    try:
        from dotenv import load_dotenv
        import requests
        
        load_dotenv()
        
        api_key = os.getenv('COINBASE_API_KEY')
        api_secret = os.getenv('COINBASE_API_SECRET')
        
        if not api_key or not api_secret:
            print_warning("Skipping API test because credentials are not set")
            return None
        
        # Simple API request to test connectivity
        base_url = "https://api.exchange.coinbase.com"
        path = "/products/ADA-USD"
        
        # Create timestamp for request
        timestamp = str(int(datetime.now().timestamp()))
        
        # Create headers
        headers = {
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'accept': 'application/json',
        }
        
        # Make request
        print_info("Testing connection to Coinbase API...")
        response = requests.get(f"{base_url}{path}", headers=headers)
        
        if response.status_code == 200:
            print_success("Successfully connected to Coinbase API")
            
            # Print some basic info about the asset
            data = response.json()
            print_info(f"Asset: {data.get('display_name', 'Unknown')}")
            print_info(f"Status: {data.get('status', 'Unknown')}")
            print_info(f"Base currency: {data.get('base_currency', 'Unknown')}")
            print_info(f"Quote currency: {data.get('quote_currency', 'Unknown')}")
            
            return True
        else:
            print_error(f"Failed to connect to Coinbase API. Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error testing API connection: {str(e)}")
        return False

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        "data",
        "data/cardano",
        "logs",
        "strategies",
        "backtest"
    ]
    
    all_dirs_exist = True
    
    print_info("Checking required directories...")
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print_success(f"Directory {directory} exists")
        else:
            print_warning(f"Directory {directory} doesn't exist. Creating it...")
            try:
                os.makedirs(directory, exist_ok=True)
                print_success(f"Created directory {directory}")
            except Exception as e:
                print_error(f"Failed to create directory {directory}: {str(e)}")
                all_dirs_exist = False
    
    return all_dirs_exist

def check_data_files():
    """Check if data files exist and can be accessed"""
    data_dir = "data/cardano"
    expected_files = [
        "ADAUSD-1m-data.csv",
        "ADAUSD-5m.csv",
        "ADAUSD-15m.csv",
        "ADAUSD-1h.csv",
        "ADAUSD-4h.csv", 
        "ADAUSD-1d.csv"
    ]
    
    found_files = []
    for file in expected_files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            found_files.append(file)
    
    if found_files:
        print_info("\nData files:")
        for file in found_files:
            file_path = os.path.join(data_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print_success(f"  {file} ({file_size:.2f} MB)")
            
            # Try to read the first few rows to ensure file is valid
            try:
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                row_count = len(pd.read_csv(file_path))
                print_info(f"    Contains {row_count} rows, columns: {', '.join(df.columns)}")
            except Exception as e:
                print_error(f"    Error reading file: {str(e)}")
        
        return True
    else:
        print_warning("\nNo data files found. Run 'python cardano_data.py' to fetch data.")
        return False

def check_cron_setup():
    """Check if cron job is set up"""
    try:
        # Get crontab
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode == 0:
            crontab = result.stdout
            if "Project Solaris" in crontab and "run_data_update.sh" in crontab:
                print_success("Cron job is set up for automatic data updates")
                
                # Find and print the schedule
                import re
                match = re.search(r'(\d+\s+\d+\s+\*\s+\*\s+\*)\s+.*run_data_update\.sh', crontab)
                if match:
                    schedule = match.group(1)
                    print_info(f"  Schedule: {schedule} (minute hour * * *)")
                
                return True
            else:
                print_warning("No cron job found for Project Solaris. Run './setup.sh' to set it up.")
                return False
        else:
            print_warning("Could not check crontab. Run './setup.sh' to set up the cron job.")
            return False
    except Exception as e:
        print_error(f"Error checking cron setup: {str(e)}")
        return False

def main():
    """Run all checks"""
    print_header(f"Project Solaris AI - Environment Test ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    print_info(f"System: {platform.system()} {platform.release()}")
    print_info(f"Working directory: {os.getcwd()}")
    
    # Run basic environment checks
    python_ok = check_python_version()
    conda_ok = check_conda_environment()
    packages_ok = check_packages()
    check_package_versions()
    env_ok = check_env_file()
    dirs_ok = check_directories()
    
    # Run advanced checks
    api_ok = None
    data_ok = None
    cron_ok = None
    
    if python_ok and packages_ok:
        if env_ok:
            print_header("API Connectivity Test")
            api_ok = test_coinbase_connectivity()
        
        print_header("Data Files Check")
        data_ok = check_data_files()
        
        print_header("Cron Job Check")
        cron_ok = check_cron_setup()
    
    # Print summary
    print_header("Environment Check Summary")
    
    if python_ok:
        print_success("Python version is compatible")
    else:
        print_error("Python version check failed")
        
    if conda_ok:
        print_success("Conda environment is properly set up")
    else:
        print_warning("Conda environment check failed")
        
    if packages_ok:
        print_success("All required packages are installed")
    else:
        print_error("Some required packages are missing")
        
    if env_ok:
        print_success(".env file is properly configured")
    else:
        print_warning(".env file check failed or incomplete")
        
    if dirs_ok:
        print_success("All required directories exist")
    else:
        print_warning("Some directories were created or are missing")
        
    if api_ok is True:
        print_success("Coinbase API connection successful")
    elif api_ok is False:
        print_error("Coinbase API connection failed")
    else:
        print_warning("Coinbase API connection not tested")
        
    if data_ok is True:
        print_success("Data files are present and readable")
    elif data_ok is False:
        print_warning("Data files check failed")
    
    if cron_ok is True:
        print_success("Cron job is set up for automatic updates")
    elif cron_ok is False:
        print_warning("Cron job is not set up")
    
    # Final verdict
    print_header("Final Assessment")
    
    critical_checks = [python_ok, packages_ok]
    important_checks = [env_ok, dirs_ok, api_ok]
    optional_checks = [conda_ok, data_ok, cron_ok]
    
    if all(check for check in critical_checks if check is not None):
        if all(check for check in important_checks if check is not None):
            if all(check for check in optional_checks if check is not None):
                print_success("🚀 Environment is fully set up and ready for trading!")
            else:
                print_warning("🔧 Environment is mostly set up but some optional components need attention.")
        else:
            print_warning("🔧 Environment has critical components ready but needs additional setup.")
    else:
        print_error("❌ Environment setup has critical issues. Please fix them before continuing.")
    
    # Suggested next steps
    print_header("Suggested Next Steps")
    
    if not packages_ok:
        print_info("Run './setup.sh' to install missing packages")
    
    if not env_ok:
        print_info("Create or edit '.env' file with your Coinbase API credentials")
    
    if not data_ok:
        print_info("Run 'python cardano_data.py' to fetch initial data")
    
    if not cron_ok:
        print_info("Run './setup.sh' to set up automated data collection")
    
    if all([python_ok, packages_ok, env_ok, dirs_ok, data_ok]):
        print_info("You're all set! You can start developing trading strategies in the 'strategies' directory.")

if __name__ == "__main__":
    main()
