#!/usr/bin/env python
'''
Fast Cardano Data Fetcher - Optimized for Historical Data
---------------------------------------------------------
This script efficiently fetches historical Cardano price data 
with smart date detection and optimized fetching strategy.
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
SYMBOL = 'ADA-USD'         # Trading pair (Cardano-USD)
TIMEFRAME = '1m'           # Base timeframe (1 minute for maximum flexibility)
DEFAULT_WEEKS = 4          # How many weeks of data to fetch by default
SAVE_DIR = 'data/cardano'  # Directory to save the data files
MAX_EMPTY_CHUNKS = 5       # Maximum consecutive empty chunks before skipping ahead

# Cardano was listed on Coinbase Pro around March 2021
DEFAULT_START_DATE = datetime.datetime(2021, 3, 19)
FIRST_DATE_WITH_DATA = datetime.datetime(2021, 3, 18)  # Known date with data

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
    parser.add_argument('--from-date', type=str, 
                        help='Starting date (YYYY-MM-DD) for data fetch')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use fast mode (optimized for historical data)')
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
        chunk_seconds = max_candles * granularity
        chunk_minutes = max(1, int(chunk_seconds / 60))
        
        # Fetch candles in chunks to avoid rate limits
        all_candles = []
        current_start = start_time
        empty_chunks_count = 0
        
        while current_start < end_time:
            # If we've hit too many empty chunks in a row, skip ahead
            if empty_chunks_count >= MAX_EMPTY_CHUNKS:
                log_warning(f"Hit {MAX_EMPTY_CHUNKS} empty chunks in a row. Skipping ahead by 1 day.")
                current_start += datetime.timedelta(days=1)
                empty_chunks_count = 0
                continue
                
            current_end = min(current_start + datetime.timedelta(minutes=chunk_minutes), end_time)
            
            log_info(f"Fetching chunk: {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
            
            params = {
                'start': current_start.isoformat(),
                'end': current_end.isoformat(),
                'granularity': str(granularity)
            }
            
            path = f'/products/{symbol}/candles'
            headers = sign_request('GET', path, api_key=api_key)
            
            try:
                response = requests.get(
                    f"{base_url}{path}",
                    params=params,
                    headers=headers,
                    timeout=30  # Add timeout to prevent hanging
                )
                
                if response.status_code != 200:
                    log_error(f"API Error: {response.status_code} - {response.text}")
                    # Wait and retry once
                    log_info("Retrying after 3 seconds...")
                    time.sleep(3)
                    
                    response = requests.get(
                        f"{base_url}{path}",
                        params=params,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        log_error(f"Failed on retry. Skipping chunk.")
                        current_start = current_end
                        empty_chunks_count += 1
                        continue
                
                candles = response.json()
                if candles and len(candles) > 0:
                    all_candles.extend(candles)
                    log_info(f"Got {len(candles)} candles")
                    empty_chunks_count = 0  # Reset empty counter on success
                else:
                    log_warning(f"No data returned for this time period")
                    empty_chunks_count += 1
                
                current_start = current_end
                time.sleep(0.25)  # Rate limit compliance - reduced for faster fetching
                
            except requests.exceptions.RequestException as e:
                log_error(f"Request error: {str(e)}")
                time.sleep(3)  # Wait for connection errors
                empty_chunks_count += 1
                # Skip to next chunk on error
                current_start = current_end
            
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

def fetch_data_by_month(start_date, end_date, symbol, timeframe, api_key):
    """
    Fetch data month by month for faster historical data loading
    """
    all_frames = []
    current_date = start_date.replace(day=1)  # Start at the first of the month
    
    # Start with a known date that has data to initialize
    if current_date < FIRST_DATE_WITH_DATA:
        log_info(f"Starting with the first known date that has data: {FIRST_DATE_WITH_DATA}")
        first_month_data = fetch_data_chunk(FIRST_DATE_WITH_DATA, 
                                            FIRST_DATE_WITH_DATA + datetime.timedelta(days=2),
                                            symbol, timeframe, api_key)
        if first_month_data is not None:
            all_frames.append(first_month_data)
            log_info(f"Got initial data from {first_month_data['datetime'].min()} to {first_month_data['datetime'].max()}")
        
        # Skip to April 2021 as the first full month
        current_date = datetime.datetime(2021, 4, 1)
    
    while current_date < end_date:
        # Calculate the first day of next month
        if current_date.month == 12:
            next_month = datetime.datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime.datetime(current_date.year, current_date.month + 1, 1)
        
        # Don't go beyond our end date
        next_month = min(next_month, end_date)
        
        log_info(f"Fetching month: {current_date.strftime('%Y-%m')}")
        
        # For recent months (last 3 months), use the more precise chunk method
        if next_month > (datetime.datetime.utcnow() - datetime.timedelta(days=90)):
            month_data = fetch_data_chunk(current_date, next_month, symbol, timeframe, api_key)
        else:
            # For historical months, just sample a few days from each month to speed things up
            # This works because we'll mainly use daily and hourly data for historical analysis
            sample_days = []
            # First day of month
            sample_days.append(current_date)
            # Middle of month
            middle = current_date + datetime.timedelta(days=14)
            sample_days.append(middle)
            # Last day of month
            last = next_month - datetime.timedelta(days=1)
            sample_days.append(last)
            
            month_frames = []
            for sample_date in sample_days:
                # Fetch a 24-hour period for each sample
                sample_end = sample_date + datetime.timedelta(hours=24)
                sample_data = fetch_data_chunk(sample_date, sample_end, symbol, timeframe, api_key)
                if sample_data is not None:
                    month_frames.append(sample_data)
            
            if month_frames:
                month_data = pd.concat(month_frames, ignore_index=True)
            else:
                month_data = None
        
        if month_data is not None and not month_data.empty:
            all_frames.append(month_data)
            log_info(f"Got data for {current_date.strftime('%Y-%m')}: {len(month_data)} rows")
        
        current_date = next_month
    
    # Combine all months
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp'])
        combined = combined.sort_values('datetime')
        return combined
    
    return None

def fetch_complete_data(start_time, end_time, symbol, timeframe, api_key, merge=False, fast_mode=True):
    """
    Fetch data for a long time period using the appropriate strategy
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
            
            # Find earliest and latest dates in existing data
            if len(existing_df) > 0:
                earliest_date = existing_df['datetime'].min()
                latest_date = existing_df['datetime'].max()
                log_info(f"Existing data range: {earliest_date} to {latest_date}")
                
                # If we already have recent data, focus on historical data
                if latest_date > (datetime.datetime.utcnow() - datetime.timedelta(days=7)):
                    log_info("Already have recent data, will focus on fetching historical data")
                    end_time = earliest_date
        except Exception as e:
            log_error(f"Error loading existing data: {str(e)}")
            existing_df = None
    
    # Choose the appropriate fetching strategy
    new_data = None
    if fast_mode and (end_time - start_time).days > 90:
        log_info("Using fast month-by-month fetching for historical data")
        new_data = fetch_data_by_month(start_time, end_time, symbol, timeframe, api_key)
    else:
        log_info("Using detailed chunk-by-chunk fetching")
        new_data = fetch_data_chunk(start_time, end_time, symbol, timeframe, api_key)
    
    # Process and combine data
    if new_data is not None and not new_data.empty:
        # Remove duplicates
        new_data = new_data.drop_duplicates(subset=['timestamp'])
        new_data = new_data.sort_values('datetime')
        
        log_info(f"Fetched {len(new_data)} new rows of data")
        log_info(f"New data range: {new_data['datetime'].min()} to {new_data['datetime'].max()}")
        
        # Merge with existing data if available
        final_df = new_data
        if existing_df is not None and not existing_df.empty:
            log_info("Merging with existing data")
            merged_df = pd.concat([existing_df, new_data], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['timestamp'])
            merged_df = merged_df.sort_values('datetime')
            final_df = merged_df
            log_info(f"Combined data has {len(final_df)} rows")
        
        # Save to CSV
        final_df.to_csv(filepath, index=False)
        log_info(f"Saved combined data to {filepath}")
        log_info(f"Final data range: {final_df['datetime'].min()} to {final_df['datetime'].max()}")
        
        return final_df
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
    
    log_info(f"Resampling {len(df)} rows of data to higher timeframes")
    
    # Make a copy with datetime as index
    df_indexed = df.copy()
    df_indexed.set_index('datetime', inplace=True)
    
    # Timeframes to generate
    timeframes = {
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',
        '4h': '4h',
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

def get_start_time(args):
    """Determine the appropriate start time based on arguments"""
    # If user specified a from-date, use that
    if args.from_date:
        try:
            start_time = datetime.datetime.strptime(args.from_date, "%Y-%m-%d")
            log_info(f"Using user-specified start date: {start_time}")
            return start_time
        except ValueError:
            log_error(f"Invalid date format: {args.from_date}. Expected YYYY-MM-DD. Using default.")
    
    # Calculate start time from years and weeks
    total_weeks = args.weeks + (args.years * 52)
    end_time = datetime.datetime.utcnow()
    calculated_start = end_time - datetime.timedelta(weeks=total_weeks)
    
    # Don't go earlier than DEFAULT_START_DATE
    if calculated_start < DEFAULT_START_DATE:
        log_info(f"Adjusted start date to when Cardano became available on Coinbase: {DEFAULT_START_DATE}")
        return DEFAULT_START_DATE
    
    return calculated_start

def main():
    """Main function"""
    args = parse_arguments()
    
    # Override globals with command line arguments
    symbol = args.symbol
    timeframe = args.timeframe
    fast_mode = args.fast
    
    log_info(f"Starting Cardano data fetcher for {symbol}")
    
    # Load API credentials
    api_key, api_secret = load_credentials()
    
    # Calculate time range
    end_time = datetime.datetime.utcnow()
    start_time = get_start_time(args)
    
    duration_days = (end_time - start_time).days
    duration_weeks = duration_days // 7
    
    log_info(f"Fetching data from {start_time} to {end_time} ({duration_days} days, ~{duration_weeks} weeks)")
    log_info(f"Using {'fast' if fast_mode else 'standard'} mode")
    
    # Fetch the data
    df = fetch_complete_data(start_time, end_time, symbol, timeframe, api_key, 
                            merge=args.merge, fast_mode=fast_mode)
    
    if df is not None:
        # Resample to other timeframes
        resample_timeframes(df, symbol)
        log_info("Data collection and processing complete!")
    else:
        log_error("Failed to fetch any data")

if __name__ == "__main__":
    main()
