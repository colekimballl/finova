import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import os
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass
from functools import wraps

# Configuration
@dataclass
class FetcherConfig:
    """Configuration for data fetcher"""
    SAVE_PATH: str = '/Users/colekimball/ztech/finova/data/backtesting'
    CACHE_PATH: str = '/Users/colekimball/ztech/finova/data/cache'
    LOG_PATH: str = '/Users/colekimball/ztech/finova/data/logs'
    CACHE_EXPIRY: int = 3600  # 1 hour in seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2
    HYPERLIQUID_BATCH_SIZE: int = 5000
    MAX_WORKERS: int = 5
    CALLS_PER_SECOND: int = 2

class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, calls_per_second: int = 2):
        self.calls_per_second = calls_per_second
        self.last_call = time.time()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < 1/self.calls_per_second:
                time.sleep(1/self.calls_per_second - elapsed)
            self.last_call = time.time()

def retry_on_exception(retries: int = 3, delay: int = 2):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator

class DataValidator:
    """Data validation utilities"""
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Validate OHLCV data"""
        if df.empty:
            return False

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False

        # Price validations
        price_checks = [
            (df['high'] >= df['low']).all(),
            (df['high'] >= df['open']).all(),
            (df['high'] >= df['close']).all(),
            (df['low'] <= df['open']).all(),
            (df['low'] <= df['close']).all(),
            (df['volume'] >= 0).all()
        ]

        return all(price_checks)

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Forward fill missing values (except volume)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # Fill missing volumes with 0
        df['volume'] = df['volume'].fillna(0)

        return df

class OptimizedStorage:
    """Optimized data storage utilities"""
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        df = df.copy()
        
        # Optimize numeric columns
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    @staticmethod
    def save_parquet(df: pd.DataFrame, filepath: str):
        """Save DataFrame to parquet with optimization"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Optimize dtypes
        df = OptimizedStorage.optimize_dtypes(df)
        
        # Save with compression
        df.to_parquet(
            filepath,
            compression='snappy',
            index=False,
            engine='pyarrow'
        )

class MarketDataFetcher:
    """Production-ready market data fetcher"""
    def __init__(self, config: Optional[FetcherConfig] = None):
        self.config = config or FetcherConfig()
        self.setup_directories()
        self.setup_logging()
        
        self.rate_limiter = RateLimiter(self.config.CALLS_PER_SECOND)
        self.validator = DataValidator()
        self.storage = OptimizedStorage()
        
        # Initialize asset lists
        self.CRYPTO_ASSETS = [
            'BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'DOGE', 'ADA', 'TRX', 'AVAX', 
            'SUI', 'TON', 'XLM', 'SHIB', 'LINK', 'HBAR', 'DOT', 'LEO', 'BCH', 
            'UNI', 'BGB', 'LTC', 'HYPE', 'PEPE'
        ]
        
        self.STOCK_ASSETS = [
            'SPY', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
            'AVGO', 'BRK-B', 'TSM', 'WMT', 'LLY', 'JPM', 'V', 'UNH', 'XOM',
            'MA', 'ORCL', 'COST'
        ]
        
        self.INTERVALS = {
            'short_term': {'interval': '1h', 'period': '60d'},
            'medium_term': {'interval': '4h', 'period': '1y'},
            'long_term': {'interval': '1d', 'period': 'max'}
        }

    def setup_directories(self):
        """Create necessary directories"""
        for path in [self.config.SAVE_PATH, self.config.CACHE_PATH, self.config.LOG_PATH]:
            os.makedirs(path, exist_ok=True)

    def setup_logging(self):
        """Configure logging"""
        log_file = os.path.join(self.config.LOG_PATH, f'market_data_{datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)

    @retry_on_exception(retries=3, delay=2)
    def fetch_hyperliquid(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch data from HyperLiquid with caching"""
        cache_file = os.path.join(self.config.CACHE_PATH, f'hyperliquid_{symbol}_{interval}.parquet')
        
        # Check cache
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < self.config.CACHE_EXPIRY:
                logger.info(f"Using cached data for {symbol}")
                return pd.read_parquet(cache_file)

        self.rate_limiter.wait()
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=60)
        
        response = requests.post(
            'https://api.hyperliquid.xyz/info',
            headers={'Content-Type': 'application/json'},
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000),
                    "limit": self.config.HYPERLIQUID_BATCH_SIZE
                }
            },
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"HyperLiquid API error: {response.status_code}")
            
        data = response.json()
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame([{
            'timestamp': datetime.utcfromtimestamp(candle['t'] / 1000),
            'open': candle['o'],
            'high': candle['h'],
            'low': candle['l'],
            'close': candle['c'],
            'volume': candle['v']
        } for candle in data])
        
        # Validate and clean data
        if not self.validator.validate_ohlcv(df):
            logger.warning(f"Data validation failed for {symbol}")
            return pd.DataFrame()
            
        df = self.validator.clean_data(df)
        
        # Cache the data
        self.storage.save_parquet(df, cache_file)
        
        return df

    @retry_on_exception(retries=3, delay=2)
    def fetch_yahoo(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with caching"""
        cache_file = os.path.join(self.config.CACHE_PATH, f'yahoo_{symbol}_{interval}.parquet')
        
        # Check cache
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < self.config.CACHE_EXPIRY:
                logger.info(f"Using cached data for {symbol}")
                return pd.read_parquet(cache_file)

        self.rate_limiter.wait()
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'timestamp', 'Datetime': 'timestamp'})
        
        # Validate and clean data
        if not self.validator.validate_ohlcv(df):
            logger.warning(f"Data validation failed for {symbol}")
            return pd.DataFrame()
            
        df = self.validator.clean_data(df)
        
        # Cache the data
        self.storage.save_parquet(df, cache_file)
        
        return df

    def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch all market data with parallel processing"""
        results = {
            'crypto': {},
            'stocks': {}
        }
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Submit all tasks
            futures = []
            
            # Crypto tasks
            for symbol in self.CRYPTO_ASSETS:
                for timeframe, params in self.INTERVALS.items():
                    futures.append(
                        executor.submit(
                            self._fetch_asset_data,
                            symbol=symbol,
                            asset_type='crypto',
                            timeframe=timeframe,
                            **params
                        )
                    )
            
            # Stock tasks
            for symbol in self.STOCK_ASSETS:
                for timeframe, params in self.INTERVALS.items():
                    futures.append(
                        executor.submit(
                            self._fetch_asset_data,
                            symbol=symbol,
                            asset_type='stocks',
                            timeframe=timeframe,
                            **params
                        )
                    )
            
            # Process results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        asset_type, symbol, timeframe, data = result
                        if symbol not in results[asset_type]:
                            results[asset_type][symbol] = {}
                        results[asset_type][symbol][timeframe] = data
                except Exception as e:
                    logger.error(f"Error processing future: {str(e)}")
        
        return results

    def _fetch_asset_data(self, symbol: str, asset_type: str, timeframe: str, 
                         interval: str, period: str) -> Optional[Tuple]:
        """Fetch data for a single asset and timeframe"""
        try:
            if asset_type == 'crypto':
                # Try HyperLiquid first
                df = self.fetch_hyperliquid(symbol, interval)
                if df.empty:
                    # Fallback to Yahoo Finance
                    df = self.fetch_yahoo(f"{symbol}-USD", period, interval)
            else:
                df = self.fetch_yahoo(symbol, period, interval)
            
            if not df.empty:
                # Save the data
                save_dir = os.path.join(self.config.SAVE_PATH, symbol, asset_type)
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f'{symbol}_{timeframe}_{timestamp}.parquet'
                filepath = os.path.join(save_dir, filename)
                
                self.storage.save_parquet(df, filepath)
                
                return asset_type, symbol, timeframe, df
                
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {str(e)}")
        
        return None

if __name__ == "__main__":
    try:
        # Initialize fetcher
        config = FetcherConfig()
        fetcher = MarketDataFetcher(config)
        
        logger.info("Starting comprehensive market data fetch...")
        start_time = time.time()
        
        # Fetch all data
        results = fetcher.fetch_all_data()
        
        # Print summary
        total_assets = 0
        total_datapoints = 0
        
        print("\n=== FETCH SUMMARY ===")
        for market_type, market_data in results.items():
            print(f"\n{market_type.upper()} MARKETS:")
            print("-" * 50)
            
            for symbol, timeframes in market_data.items():
                print(f"\n{symbol}:")
                asset_total = 0
                
                for timeframe, df in timeframes.items():
                    rows = len(df) if not df.empty else 0
                    asset_total += rows
                    total_datapoints += rows
                    print(f"  {timeframe:12} : {rows:7,} candles | "
                          f"Range: {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}")
                
                print(f"  {'TOTAL':12} : {asset_total:7,} candles")
                total_assets += 1
        
        # Print execution statistics
        execution_time = time.time() - start_time
        print("\n=== EXECUTION STATISTICS ===")
        print(f"Total Assets Processed : {total_assets}")
        print(f"Total Data Points     : {total_datapoints:,}")
        print(f"Execution Time        : {execution_time:.2f} seconds")
        print(f"Average Time/Asset    : {execution_time/total_assets:.2f} seconds")
        print(f"Data Points/Second    : {total_datapoints/execution_time:,.2f}")
        
        logger.info("Market data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
