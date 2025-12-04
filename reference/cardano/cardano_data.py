#!/usr/bin/env python
'''
Enhanced Cardano Data Fetcher
--------------------------
This script fetches historical Cardano price data from Coinbase
with support for long-term data collection and improved argument handling.
'''

import pandas as pd
import numpy as np
import datetime
import os
import time
import json
import logging
import sys
import argparse
from pathlib import Path
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataFetcher")

# ====== Configuration ======
SYMBOL = 'ADA-USD'        # Trading pair (Cardano-USD)
TIMEFRAME = '1m'          # Base timeframe (1 minute for maximum flexibility)
DEFAULT_WEEKS = 4         # How many weeks of data to fetch by default
MAX_WEEKS_PER_REQUEST = 4  # Maximum weeks to fetch in a single run to avoid timeout
SAVE_DIR = 'data/cardano'  # Directory to save the data files

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

def log_info(message):
    """Log info message"""
    logger.info(message)
    
def log_error(message):
    """Log error message"""
    logger.error(message)
    
def log_warning(message):
    """Log warning message"""
    logger.warning(message)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fetch Cardano historical data')
    parser.add_argument('--weeks', type=int, default=DEFAULT_WEEKS,
                        help=f'Number of weeks of historical data to fetch (default: {DEFAULT_WEEKS})')
    parser.add_argument('--years', type=int, default=0,
                        help='Number of years of historical data to fetch')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME,
                        help=f'Base timeframe to fetch (default: {TIMEFRAME})')
    parser.add_argument('--symbol', type=str, default=SYMBOL,
                        help=f'Trading pair to fetch (default: {SYMBOL})')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with existing data if available')
    return parser.parse_args()

# Load environment variables
def load_credentials():
    log_info(f"Loading .env file from: {os.getcwd()}")
    load_dotenv()

    # Check API credentials
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')

    if not api_key or not api_secret:
        log_error("API credentials not found in .env file")
        sys.exit(1)

    log_info("API credentials loaded successfully")
    return api_key, api_secret

def sign_request(method, path, body='', timestamp=None, api_key=None):
    """Sign a request using the API secret"""
    timestamp = timestamp or str(int(time.time()))
    
    # Create headers
    headers = {
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'accept': 'application/json',
        'content-type': 'application/json',
    }
    
    return headers

def timeframe_to_granularity(timeframe):
    """Convert timeframe to granularity in seconds"""
    if 'm' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60
    elif 'h' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60 * 60
    elif 'd' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 24 * 60 * 60

def fetch_data_chunk(start_time, end_time, symbol, timeframe, api_key):
    """Fetch historical data for a specific time range"""
    log_info(f"Fetching data from {start_time} to {end_time}")
    
    try:
        base_url = "https://api.exchange.coinbase.com"
        granularity = timeframe_to_granularity(timeframe)
        
        # Coinbase limit is 300 candles per request
        max_candles = 300
        chunk_minutes = max(1, int((max_candles * granularity) / 60))
        
        # Fetch candles in chunks to avoid rate limits
        all_candles = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + datetime.timedelta(minutes=chunk_minutes), end_time)
            
            log_info(f"Fetching chunk: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
            
            params = {
                'start': current_start.isoformat(),
                'end': current_end.isoformat(),
                'granularity': str(granularity)
            }
            
            path = f'/products/{symbol}/candles'
            headers = sign_request('GET', path, api_key=api_key)
            
            response = requests.get(
                f"{base_url}{path}",
                params=params,
                headers=headers
            )
            
            if response.status_code != 200:
                log_error(f"API Error: {response.status_code} - {response.text}")
                # Wait and retry once
                log_info("Retrying after 5 seconds...")
                time.sleep(5)
                
                response = requests.get(
                    f"{base_url}{path}",
                    params=params,
                    headers=headers
                )
                
                if response.status_code != 200:
                    log_error(f"Failed on retry. Skipping chunk.")
                    current_start = current_end
                    continue
            
            candles = response.json()
            if candles:
                all_candles.extend(candles)
                log_info(f"Got {len(candles)} candles")
            else:
                log_warning(f"No data returned for this time period")
            
            current_start = current_end
            time.sleep(0.5)  # Rate limit compliance
            
        log_info(f"Successfully fetched {len(all_candles)} candles for this chunk!")
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles)
            df.columns = ['timestamp', 'low', 'high', 'open', 'close', 'volume']
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('datetime')
            return df
        else:
            log_warning("No data fetched for this chunk")
            return None
            
    except Exception as e:
        log_error(f"Error fetching data chunk: {str(e)}")
        return None

def fetch_complete_data(start_time, end_time, symbol, timeframe, api_key, merge=False):
    """
    Fetch data for a long time period by breaking it into manageable chunks
    """
    # Check if we should load existing data
    filename = f"{symbol.replace('-', '')}-{timeframe}-data.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    
    existing_df = None
    if merge and os.path.exists(filepath):
        try:
            log_info(f"Loading existing data from {filepath}")
            existing_df = pd.read_csv(filepath)
            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
            log_info(f"Loaded {len(existing_df)} rows of existing data")
        except Exception as e:
            log_error(f"Error loading existing data: {str(e)}")
            existing_df = None
    
    # Calculate chunks of MAX_WEEKS_PER_REQUEST each
    total_days = (end_time - start_time).days
    chunk_days = min(total_days, MAX_WEEKS_PER_REQUEST * 7)
    
    all_data_frames = []
    current_start = start_time
    
    while current_start < end_time:
        current_end = min(current_start + datetime.timedelta(days=chunk_days), end_time)
        
        log_info(f"Fetching chunk from {current_start} to {current_end} ({(current_end - current_start).days} days)")
        df_chunk = fetch_data_chunk(current_start, current_end, symbol, timeframe, api_key)
        
        if df_chunk is not None and not df_chunk.empty:
            all_data_frames.append(df_chunk)
            
        current_start = current_end
    
    # Combine all chunks
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Remove duplicates based on timestamp
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        
        # Sort by datetime
        combined_df = combined_df.sort_values('datetime')
        
        # Merge with existing data if available
        if existing_df is not None and not existing_df.empty:
            log_info("Merging with existing data")
            merged_df = pd.concat([existing_df, combined_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['timestamp'])
            merged_df = merged_df.sort_values('datetime')
            combined_df = merged_df
        
        # Save to CSV
        combined_df.to_csv(filepath, index=False)
        log_info(f"Saved combined data with {len(combined_df)} rows to {filepath}")
        
        return combined_df
    elif existing_df is not None:
        log_info("No new data fetched, using existing data")
        return existing_df
    else:
        log_warning("No data fetched")
        return None

def resample_timeframes(df, symbol):
    """Resample data to higher timeframes"""
    if df is None or df.empty:
        log_warning("No data to resample")
        return
    
    # Make a copy with datetime as index
    df_indexed = df.copy()
    df_indexed.set_index('datetime', inplace=True)
    
    # Timeframes to generate
    timeframes = {
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',  # Changed from '1H' to '1h' to avoid deprecation warning
        '4h': '4h',  # Changed from '4H' to '4h' to avoid deprecation warning
        '1d': 'D'
    }
    
    for tf_name, tf_rule in timeframes.items():
        log_info(f"Resampling to {tf_name} timeframe")
        
        try:
            resampled = df_indexed.resample(tf_rule).agg({
                'timestamp': 'first',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Remove rows with NaN values
            resampled = resampled.dropna()
            
            # Reset index to include datetime as a column
            resampled = resampled.reset_index()
            
            # Save to file
            filename = f"{symbol.replace('-', '')}-{tf_name}.csv"
            filepath = os.path.join(SAVE_DIR, filename)
            resampled.to_csv(filepath, index=False)
            log_info(f"Saved {tf_name} data with {len(resampled)} rows to {filepath}")
        except Exception as e:
            log_error(f"Error resampling to {tf_name}: {str(e)}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Override globals with command line arguments
    symbol = args.symbol
    timeframe = args.timeframe
    
    # Calculate total weeks from years and weeks arguments
    total_weeks = args.weeks + (args.years * 52)
    
    log_info(f"Starting Cardano data fetcher for {symbol} with {total_weeks} weeks of data")
    
    # Load API credentials
    api_key, api_secret = load_credentials()
    
    # Calculate time range
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(weeks=total_weeks)
    
    log_info(f"Fetching data from {start_time} to {end_time} ({total_weeks} weeks)")
    
    # Fetch the data
    df = fetch_complete_data(start_time, end_time, symbol, timeframe, api_key, merge=args.merge)
    
    if df is not None:
        # Resample to other timeframes
        resample_timeframes(df, symbol)
        log_info("Data collection and processing complete!")
    else:
        log_error("Failed to fetch any data")

if __name__ == "__main__":
    main()
