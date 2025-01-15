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
    """Reliable market data fetcher focusing on Yahoo Finance"""
    
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
            'hourly': {'interval': '1h', 'period': '730d'},  # 2 years of hourly data
            'weekly': {'interval': '1wk', 'period': 'max'}
        }
        
        # Create directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for data storage"""
        directories = ['stocks', 'crypto']
        for market in directories:
            path = os.path.join(self.base_path, market)
            os.makedirs(path, exist_ok=True)

    def fetch_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol and timeframe"""
        try:
            logger.info(f"Fetching {timeframe} data for {symbol}")
            
            # Get timeframe parameters
            params = self.TIMEFRAMES[timeframe]
            
            # Initialize ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch data
            df = ticker.history(
                period=params['period'],
                interval=params['interval'],
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return None
                
            # Process the dataframe
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure consistent column naming
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'date'})
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return None

    def save_data(self, df: pd.DataFrame, symbol: str, market_type: str, timeframe: str):
        """Save data to parquet format"""
        if df is None or df.empty:
            return
            
        try:
            # Create directory for symbol
            symbol_dir = os.path.join(self.base_path, market_type, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}.parquet"
            filepath = os.path.join(symbol_dir, filename)
            
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {symbol} {timeframe} data to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving {symbol}: {str(e)}")

    def fetch_market_data(self, max_workers: int = 5):
        """Fetch data for all symbols with parallel processing"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process stocks
            for symbol in self.STOCKS:
                for timeframe in self.TIMEFRAMES:
                    executor.submit(self._process_symbol, symbol, 'stocks', timeframe)
            
            # Process crypto
            for symbol in self.CRYPTO:
                for timeframe in self.TIMEFRAMES:
                    executor.submit(self._process_symbol, symbol, 'crypto', timeframe)

    def _process_symbol(self, symbol: str, market_type: str, timeframe: str):
        """Process a single symbol"""
        try:
            # Fetch data
            df = self.fetch_data(symbol, timeframe)
            
            # Save if successful
            if df is not None and not df.empty:
                self.save_data(df, symbol, market_type, timeframe)
                
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")

def main():
    """Main execution function"""
    try:
        start_time = time.time()
        logger.info("Starting market data collection...")
        
        # Initialize fetcher
        fetcher = MarketDataFetcher()
        
        # Fetch all data
        fetcher.fetch_market_data()
        
        # Print summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info("Data collection completed!")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        # Verify data
        success_count = 0
        total_files = 0
        total_size = 0
        
        for market in ['stocks', 'crypto']:
            market_path = os.path.join(fetcher.base_path, market)
            if os.path.exists(market_path):
                for root, dirs, files in os.walk(market_path):
                    for file in files:
                        if file.endswith('.parquet'):
                            total_files += 1
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                            success_count += 1
        
        print("\n=== Collection Summary ===")
        print(f"Successfully processed: {success_count} files")
        print(f"Total data size: {total_size:.2f} MB")
        print(f"Average processing time per file: {execution_time/total_files:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
