#!/usr/bin/env python
'''
Project Solaris - Enhanced Coinbase Data Fetcher
-----------------------------------------------
This script fetches historical and current data for Cardano (ADA) from Coinbase.
It maintains a complete history in 1-minute resolution and can be run via cron
for automatic daily updates.

Improvements:
- Better error handling and logging
- Support for multiple timeframes conversion from 1-minute data
- More robust date handling
- Progress tracking for longer operations
'''

import pandas as pd
import datetime
import os
import time
import hmac
import hashlib
import base64
import json
import logging
import sys
from pathlib import Path
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import pandas_ta as ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/coinbase_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CoinbaseDataFetcher")

# ====== Configuration ======
SYMBOL = 'ADA-USD'        # Trading pair (Cardano-USD)
TIMEFRAME = '1m'          # Base timeframe (1 minute for maximum flexibility)
DEFAULT_WEEKS = 52        # How many weeks of data to fetch by default
SAVE_DIR = 'data/cardano'  # Directory to save the data files

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

def log_info(message):
    """Log info message to both console and file"""
    logger.info(message)
    
def log_error(message):
    """Log error message to both console and file"""
    logger.error(message)
    
def log_warning(message):
    """Log warning message to both console and file"""
    logger.warning(message)

# Load environment variables
log_info(f"Looking for .env file in: {os.getcwd()}")

# Get the project root directory
project_root = Path(__file__).parent
env_path = project_root / '.env'

log_info(f".env file exists: {'✅' if env_path.exists() else '❌'}")

# Load environment variables
load_dotenv()

# Debug prints for API credentials (without revealing them)
api_key = os.getenv('COINBASE_API_KEY')
api_secret = os.getenv('COINBASE_API_SECRET')
log_info("API Key loaded: " + ("✅" if api_key else "❌"))
log_info("API Secret loaded: " + ("✅" if api_secret else "❌"))

if not api_key or not api_secret:
    log_error("API credentials not found in .env file")
    log_error("Make sure your .env file exists and contains:")
    log_error("COINBASE_API_KEY=organizations/{org_id}/apiKeys/{key_id}")
    log_error("COINBASE_API_SECRET=your-secret-key")
    sys.exit(1)

def sign_request(method, path, body='', timestamp=None):
    """Sign a request using the API secret"""
    timestamp = timestamp or str(int(time.time()))
    
    # Remove the '/api/v3/brokerage' prefix from path for signing
    if path.startswith('/api/v3/brokerage'):
        path = path[len('/api/v3/brokerage'):]
    
    # Create the message to sign
    message = f"{timestamp}{method}{path}{body}"
    
    try:
        log_info(f"Signing request for: {method} {path}")
        
        # Create headers
        headers = {
            'CB-ACCESS-KEY': api_key,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        
        return headers
        
    except Exception as e:
        log_error(f"Error generating signature: {str(e)}")
        raise

def timeframe_to_granularity(timeframe):
    """Convert timeframe to granularity in seconds"""
    if 'm' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60
    elif 'h' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60 * 60
    elif 'd' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 24 * 60 * 60

def get_latest_data_time():
    """Get the latest timestamp from existing data files"""
    try:
        # Get master data file if it exists
        master_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-master.csv')
        if os.path.exists(master_file):
            df = pd.read_csv(master_file)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df['datetime'].max()
    except Exception as e:
        log_error(f"Error reading latest data time: {e}")
    
    # If no existing data or error, return None (will trigger full historical load)
    return None

def fetch_historical_data(start_time, end_time):
    """Fetch historical data for a specific time range"""
    log_info(f"Fetching data from {start_time} to {end_time}")
    
    try:
        base_url = "https://api.exchange.coinbase.com"
        granularity = timeframe_to_granularity(TIMEFRAME)
        
        # Calculate appropriate chunk size based on granularity
        # Coinbase limit is 300 candles per request
        max_candles = 300
        chunk_minutes = max(1, int((max_candles * granularity) / 60))  # Convert to minutes for 1m data
        
        # Fetch candles in chunks to avoid rate limits
        all_candles = []
        current_start = start_time
        chunk_count = 0
        total_chunks = int((end_time - start_time).total_seconds() / (chunk_minutes * 60)) + 1
        
        log_info(f"Will fetch approximately {total_chunks} data chunks")
        
        while current_start < end_time:
            current_end = min(current_start + datetime.timedelta(minutes=chunk_minutes), end_time)
            chunk_count += 1
            
            log_info(f"Fetching chunk {chunk_count}/{total_chunks}: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
            
            params = {
                'start': current_start.isoformat(),
                'end': current_end.isoformat(),
                'granularity': str(granularity)
            }
            
            path = f'/products/{SYMBOL}/candles'
            headers = sign_request('GET', path)
            
            response = requests.get(
                f"{base_url}{path}",
                params=params,
                headers=headers
            )
            
            if response.status_code != 200:
                log_error(f"API Error: {response.status_code} - {response.text}")
                # Wait and retry once for this chunk
                log_info(f"Retrying after 5 seconds...")
                time.sleep(5)
                
                response = requests.get(
                    f"{base_url}{path}",
                    params=params,
                    headers=headers
                )
                
                if response.status_code != 200:
                    log_error(f"Failed to fetch chunk after retry. Continuing to next chunk.")
                    current_start = current_end
                    continue
                    
            candles = response.json()
            all_candles.extend(candles)
            
            log_info(f"Got {len(candles)} candles")
            
            current_start = current_end
            time.sleep(0.5)  # Rate limit compliance
            
        log_info(f"Successfully fetched {len(all_candles)} candles!")
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles)
            df.columns = ['timestamp', 'low', 'high', 'open', 'close', 'volume']
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('datetime')
            return df
        else:
            log_warning("No new data found for this time period")
            return pd.DataFrame()
            
    except Exception as e:
        log_error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def update_master_file(new_data):
    """Update the master data file with new data"""
    master_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-master.csv')
    
    try:
        # If master file exists, read it and append new data
        if os.path.exists(master_file):
            existing_data = pd.read_csv(master_file)
            existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
            
            # Combine and remove duplicates
            combined = pd.concat([existing_data, new_data])
            combined = combined.drop_duplicates(subset=['datetime'])
            combined = combined.sort_values('datetime')
            
            log_info(f"Master file updated: {len(existing_data)} old rows + {len(new_data)} new rows = {len(combined)} total rows")
            
            # Save updated master file
            combined.to_csv(master_file, index=False)
            return combined
        else:
            # If no master file exists, create it
            new_data.to_csv(master_file, index=False)
            log_info(f"Created new master file with {len(new_data)} rows")
            return new_data
            
    except Exception as e:
        log_error(f"Error updating master file: {str(e)}")
        return pd.DataFrame()

def create_daily_snapshot(data, snapshot_date):
    """Create a daily snapshot file"""
    date_str = snapshot_date.strftime('%Y-%m-%d')
    snapshot_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-{date_str}.csv')
    
    # Filter data for the specific date
    day_data = data[data['datetime'].dt.date == snapshot_date.date()]
    
    if not day_data.empty:
        day_data.to_csv(snapshot_file, index=False)
        log_info(f"Created daily snapshot for {date_str} with {len(day_data)} rows")
    else:
        log_warning(f"No data available for {date_str}, snapshot not created")

def resample_to_higher_timeframes(data, timeframes=['5m', '15m', '1h', '4h', '1d']):
    """Resample 1-minute data to higher timeframes"""
    log_info(f"Resampling data to higher timeframes: {timeframes}")
    
    if data.empty:
        log_warning("No data to resample")
        return
        
    # Ensure datetime is the index for resampling
    if 'datetime' in data.columns:
        data = data.set_index('datetime')
    
    # Mapping of timeframe strings to pandas resample rules
    rule_map = {
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': 'H',
        '4h': '4H',
        '1d': 'D'
    }
    
    for tf in timeframes:
        if tf not in rule_map:
            log_warning(f"Unsupported timeframe: {tf}")
            continue
            
        rule = rule_map[tf]
        
        # Resample the data
        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'timestamp': 'first'  # Keep the original timestamp
        })
        
        # Reset index to include datetime as a column
        resampled = resampled.reset_index()
        
        # Save to file
        output_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{tf}.csv')
        resampled.to_csv(output_file, index=False)
        
        log_info(f"Created {tf} timeframe with {len(resampled)} rows")

def add_technical_indicators(data, save=True):
    """Add technical indicators to the dataset"""
    log_info("Adding technical indicators")
    
    if data.empty:
        log_warning("No data to add indicators to")
        return data
        
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Moving Averages
    log_info("Adding SMAs")
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    
    log_info("Adding EMAs")
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)
    
    # MACD
    log_info("Adding MACD")
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # RSI
    log_info("Adding RSI")
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # Bollinger Bands
    log_info("Adding Bollinger Bands")
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    
    if save:
        # Save enhanced data
        output_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-indicators.csv')
        df.to_csv(output_file, index=False)
        log_info(f"Saved data with indicators to {output_file}")
    
    return df

def fetch_all_data(weeks=DEFAULT_WEEKS):
    """Main function to fetch all data"""
    log_info(f"Starting Cardano data fetcher for {weeks} weeks of history")
    
    # Determine time range to fetch
    end_time = datetime.datetime.utcnow()
    latest_time = get_latest_data_time()
    
    if latest_time is None:
        # No existing data, fetch historical data
        log_info(f"No existing data found. Fetching {weeks} weeks of historical data...")
        start_time = end_time - datetime.timedelta(weeks=weeks)
    else:
        # We have data, just fetch the new data since last update
        # Add a small buffer (1 hour) to handle any potential gaps
        log_info(f"Existing data found with last timestamp: {latest_time}")
        start_time = latest_time - datetime.timedelta(hours=1)
    
    # Fetch the data
    new_data = fetch_historical_data(start_time, end_time)
    
    if not new_data.empty:
        # Update the master file
        all_data = update_master_file(new_data)
        
        # Resample to higher timeframes
        resample_to_higher_timeframes(all_data)
        
        # Add technical indicators
        add_technical_indicators(all_data)
        
        # Create daily snapshot for today
        today = datetime.datetime.utcnow().date()
        create_daily_snapshot(all_data, today)
        
        log_info(f"Data refresh completed successfully!")
        return True
    else:
        log_warning("No new data fetched")
        return False

def show_data_info():
    """Show information about the current dataset"""
    master_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-master.csv')
    
    if not os.path.exists(master_file):
        log_info("No master data file found")
        return
        
    try:
        df = pd.read_csv(master_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Basic statistics
        date_range = df['datetime'].max() - df['datetime'].min()
        total_days = date_range.total_seconds() / (24 * 60 * 60)
        
        log_info("\nData Statistics:")
        log_info(f"Total records: {len(df)}")
        log_info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()} ({total_days:.1f} days)")
        log_info(f"Theoretical 1-minute candles in range: {int(total_days * 24 * 60)}")
        log_info(f"Actual candles: {len(df)}")
        log_info(f"Data completion: {len(df) / (total_days * 24 * 60) * 100:.2f}%")
        
        # Price statistics
        log_info("\nPrice Statistics:")
        log_info(f"All-time high: ${df['high'].max():.4f}")
        log_info(f"All-time low: ${df['low'].min():.4f}")
        log_info(f"Current price: ${df['close'].iloc[-1]:.4f}")
        
        # Check for missing data
        daily_counts = df.groupby(df['datetime'].dt.date).count()['close']
        missing_days = daily_counts[daily_counts < 1000].index.tolist()
        
        if missing_days:
            log_warning("\nDays with significant missing data:")
            for day in missing_days[:10]:  # Show first 10 only
                log_warning(f"  {day}: {daily_counts.loc[day]} candles")
            
            if len(missing_days) > 10:
                log_warning(f"  ...and {len(missing_days) - 10} more days")
                
    except Exception as e:
        log_error(f"Error reading data info: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cardano Data Fetcher')
    parser.add_argument('--weeks', type=int, default=DEFAULT_WEEKS, 
                        help=f'Number of weeks of historical data to fetch (default: {DEFAULT_WEEKS})')
    parser.add_argument('--info', action='store_true', 
                        help='Show information about the current dataset')
    parser.add_argument('--indicators', action='store_true',
                        help='Add technical indicators to existing data without fetching new data')
    parser.add_argument('--resample', action='store_true',
                        help='Resample existing 1-minute data to higher timeframes without fetching new data')
    
    args = parser.parse_args()
    
    if args.info:
        show_data_info()
    elif args.indicators:
        master_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-master.csv')
        if os.path.exists(master_file):
            log_info("Adding indicators to existing data...")
            data = pd.read_csv(master_file)
            data['datetime'] = pd.to_datetime(data['datetime'])
            add_technical_indicators(data)
        else:
            log_error("No master data file found. Please fetch data first.")
    elif args.resample:
        master_file = os.path.join(SAVE_DIR, f'{SYMBOL.replace("-", "")}-{TIMEFRAME}-master.csv')
        if os.path.exists(master_file):
            log_info("Resampling existing data to higher timeframes...")
            data = pd.read_csv(master_file)
            data['datetime'] = pd.to_datetime(data['datetime'])
            resample_to_higher_timeframes(data)
        else:
            log_error("No master data file found. Please fetch data first.")
    else:
        fetch_all_data(weeks=args.weeks)
