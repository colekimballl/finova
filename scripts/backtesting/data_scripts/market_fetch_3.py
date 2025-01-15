import pandas as pd
import yfinance as yf
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self, base_path: str = '/Users/colekimball/ztech/finova/data/backtesting'):
        self.base_path = base_path
        
        # Define assets
        self.STOCKS = [
            'SPY', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
            'AVGO', 'BRK-B', 'TSM', 'WMT', 'LLY', 'JPM', 'V', 'UNH', 'XOM',
            'MA', 'ORCL', 'COST'
        ]
        
        self.CRYPTO = [
            'BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD', 'DOGE-USD', 
            'ADA-USD', 'TRX-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'UNI-USD',
            'LTC-USD'
        ]
        
        # Yahoo Finance compatible intervals and periods
        self.TIMEFRAMES = {
            'daily': {'interval': '1d', 'period': 'max'},
            'hourly': {'interval': '1h', 'period': '7d'},  # Maximum allowed for 1h
            'weekly': {'interval': '1wk', 'period': 'max'}
        }
        
        # Create directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for data storage"""
        for market in ['stocks', 'crypto']:
            path = os.path.join(self.base_path, market)
            os.makedirs(path, exist_ok=True)

    def fetch_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol and timeframe"""
        try:
            params = self.TIMEFRAMES[timeframe]
            ticker = yf.Ticker(symbol)
            
            df = ticker.history(
                period=params['period'],
                interval=params['interval'],
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data received for {symbol} ({timeframe})")
                return None
            
            # Process the dataframe
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'date'})
            
            logger.info(f"Fetched {len(df)} rows for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} ({timeframe}): {str(e)}")
            return None

    def save_data(self, df: pd.DataFrame, symbol: str, market_type: str, timeframe: str):
        """Save data to parquet format"""
        if df is None or df.empty:
            return
            
        try:
            # Create directory for symbol
            symbol_dir = os.path.join(self.base_path, market_type, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save file (without timestamp to enable overwriting)
            filename = f"{symbol}_{timeframe}.parquet"
            filepath = os.path.join(symbol_dir, filename)
            
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} rows for {symbol} ({timeframe})")
            
        except Exception as e:
            logger.error(f"Error saving {symbol}: {str(e)}")

    def process_symbol(self, symbol: str, market_type: str):
        """Process all timeframes for a single symbol"""
        for timeframe in self.TIMEFRAMES:
            try:
                logger.info(f"Processing {symbol} - {timeframe}")
                df = self.fetch_data(symbol, timeframe)
                if df is not None:
                    self.save_data(df, symbol, market_type, timeframe)
            except Exception as e:
                logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")

    def fetch_all_data(self, max_workers: int = 5):
        """Fetch all market data with parallel processing"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process stocks
            stock_futures = [
                executor.submit(self.process_symbol, symbol, 'stocks')
                for symbol in self.STOCKS
            ]
            
            # Process crypto
            crypto_futures = [
                executor.submit(self.process_symbol, symbol, 'crypto')
                for symbol in self.CRYPTO
            ]
            
            # Wait for completion
            for future in stock_futures + crypto_futures:
                future.result()

def main():
    try:
        start_time = time.time()
        logger.info("Starting market data collection...")
        
        # Initialize and run fetcher
        fetcher = MarketDataFetcher()
        fetcher.fetch_all_data()
        
        # Print summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify data
        success_count = 0
        total_size = 0
        
        print("\n=== Collection Summary ===")
        for market in ['stocks', 'crypto']:
            print(f"\n{market.upper()} Market:")
            market_path = os.path.join(fetcher.base_path, market)
            if os.path.exists(market_path):
                for symbol in os.listdir(market_path):
                    symbol_path = os.path.join(market_path, symbol)
                    if os.path.isdir(symbol_path):
                        files = [f for f in os.listdir(symbol_path) if f.endswith('.parquet')]
                        if files:
                            print(f"{symbol}:")
                            for file in files:
                                file_path = os.path.join(symbol_path, file)
                                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                                df = pd.read_parquet(file_path)
                                print(f"  - {file}: {len(df)} rows, {file_size:.2f}MB")
                                success_count += 1
                                total_size += file_size
        
        print(f"\nTotal files: {success_count}")
        print(f"Total size: {total_size:.2f}MB")
        print(f"Execution time: {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
