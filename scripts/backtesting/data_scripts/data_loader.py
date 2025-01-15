# data_loader.py
import pandas as pd
import os
from typing import Dict, List, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataLoader:
    def __init__(self, base_path: str = '/Users/colekimball/ztech/finova/data/backtesting/cleaned'):
        self.base_path = base_path
        
    def load_single_asset(self, symbol: str, timeframe: str = 'daily', market: str = 'stocks') -> pd.DataFrame:
        """Load data for a single symbol"""
        try:
            filepath = os.path.join(self.base_path, market, f"{symbol}_{timeframe}.parquet")
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
                
            df = pd.read_parquet(filepath)
            df['symbol'] = symbol
            logger.info(f"Loaded {symbol} {timeframe}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            return None
    
    def load_market(self, market: str = 'stocks', timeframe: str = 'daily') -> Dict[str, pd.DataFrame]:
        """Load all assets for a specific market"""
        market_path = os.path.join(self.base_path, market)
        if not os.path.exists(market_path):
            logger.error(f"Market path not found: {market_path}")
            return {}
            
        # Get unique symbols
        symbols = set()
        for file in os.listdir(market_path):
            if file.endswith(f'_{timeframe}.parquet'):
                symbols.add(file.split('_')[0])
        
        data = {}
        for symbol in sorted(symbols):
            df = self.load_single_asset(symbol, timeframe, market)
            if df is not None:
                data[symbol] = df
        
        return data

    def get_merged_data(self, market: str = 'stocks', timeframe: str = 'daily') -> pd.DataFrame:
        """Get merged data for all assets in a market"""
        data_dict = self.load_market(market, timeframe)
        if not data_dict:
            logger.error(f"No data loaded for {market} {timeframe}")
            return pd.DataFrame()
            
        # Merge all dataframes
        merged_data = pd.concat(data_dict.values(), axis=0)
        
        # Sort by date and symbol
        merged_data = merged_data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        logger.info(f"Merged data shape: {merged_data.shape}")
        return merged_data

    def print_data_info(self, data: Dict[str, pd.DataFrame]):
        """Print information about loaded data"""
        print(f"\nLoaded {len(data)} assets:")
        for symbol, df in data.items():
            print(f"{symbol:6}: {len(df):6,} rows | "
                  f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
