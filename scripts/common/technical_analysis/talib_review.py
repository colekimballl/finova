# ta_lib_review.py

import pandas as pd
import talib as ta
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
import warnings
import yaml
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from termcolor import cprint
import numpy as np

# Setup logging with both file and console output
log_dir = Path('/Users/colekimball/ztech/finova/logs')
log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
log_file = log_dir / f'ta_lib_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Setup file handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Setup console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class DataSource(ABC):
    """Abstract base class for different data sources"""
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Get data from source"""
        pass

class CSVDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

class TechnicalAnalysisTA:
    """Technical Analysis class for calculating and managing indicators using TA-Lib"""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config path"""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.start_time = time.time()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration for technical indicators"""
        return {
            'trend_indicators': {
                'sma_period': 20,
                'ema_period': 20,
                'tema_period': 20,
                'wma_period': 20,
                'kama_period': 20,
                'trima_period': 20
            },
            'momentum_indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'mom_period': 10,
                'roc_period': 10,
                'willr_period': 14
            },
            'volatility_indicators': {
                'atr_period': 14,
                'bbands_period': 20,
                'bbands_dev': 2
            },
            'volume_indicators': {
                'mfi_period': 14
            }
        }

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return TechnicalAnalysisTA._default_config()

    def setup_environment(self) -> None:
        """Initialize and verify environment"""
        logger.info("=== TA-Lib Technical Analysis Process Starting ===")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Pandas Version: {pd.__version__}")
        logger.info(f"TA-Lib Version: {ta.__version__}")
        logger.info(f"Python Path: {sys.executable}")
        logger.info(f"Configuration: {self.config}")

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean the dataset"""
        logger.info(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows of data")
            
            # Clean data by removing first two rows and resetting index
            df = df.iloc[2:].reset_index(drop=True)
            logger.info("Removed header rows")
            
            # Convert price columns to numeric types
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    raise ValueError(f"Missing expected column: {col}")
                    
            # Data quality checks
            self._check_data_quality(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Perform data quality checks"""
        # Check for null values
        null_counts = df[['Close', 'High', 'Low', 'Open', 'Volume']].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values in columns: \n{null_counts[null_counts > 0]}")
        
        # Check for negative values
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if (df[col] < 0).any():
                logger.warning(f"Found negative values in {col}")
        
        # Check for price inconsistencies
        if (df['High'] < df['Low']).any():
            logger.error("Found High prices lower than Low prices")
        
        # Check for zero prices
        zero_prices = (df[['Close', 'High', 'Low', 'Open']] == 0).sum()
        if zero_prices.any():
            logger.warning(f"Found zero prices: \n{zero_prices[zero_prices > 0]}")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators using TA-Lib"""
        logger.info("Starting technical indicator calculations using TA-Lib")
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Calculate TA-Lib Indicators
                self._calculate_ta_lib_indicators(df)
                
                # Log any warnings captured during indicator calculations
                if len(w) > 0:
                    for warning in w:
                        logger.warning(f"Calculation warning: {warning.message}")
                        
            logger.info("TA-Lib technical indicator calculations completed successfully.")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _calculate_ta_lib_indicators(self, df: pd.DataFrame) -> None:
        """Calculate technical indicators using TA-Lib"""
        logger.info("Calculating TA-Lib technical indicators...")
        c = self.config['trend_indicators']
        df['sma'] = ta.SMA(df['Close'], timeperiod=c['sma_period'])
        df['ema'] = ta.EMA(df['Close'], timeperiod=c['ema_period'])
        df['tema'] = ta.TEMA(df['Close'], timeperiod=c['tema_period'])
        df['wma'] = ta.WMA(df['Close'], timeperiod=c['wma_period'])
        df['kama'] = ta.KAMA(df['Close'], timeperiod=c['kama_period'])
        df['trima'] = ta.TRIMA(df['Close'], timeperiod=c['trima_period'])
        
        c = self.config['momentum_indicators']
        df['rsi'] = ta.RSI(df['Close'], timeperiod=c['rsi_period'])
        df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(
            df['Close'], 
            fastperiod=c['macd_fast'],
            slowperiod=c['macd_slow'],
            signalperiod=c['macd_signal']
        )
        df['mom'] = ta.MOM(df['Close'], timeperiod=c['mom_period'])
        df['roc'] = ta.ROC(df['Close'], timeperiod=c['roc_period'])
        df['willr'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=c['willr_period'])
        
        c = self.config['volatility_indicators']
        df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=c['atr_period'])
        df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(
            df['Close'],
            timeperiod=c['bbands_period'],
            nbdevup=c['bbands_dev'],
            nbdevdn=c['bbands_dev']
        )
        
        c = self.config['volume_indicators']
        df['obv'] = ta.OBV(df['Close'], df['Volume'])
        df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=c['mfi_period'])
        
        logger.info("TA-Lib technical indicators calculated successfully.")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate and log summary statistics"""
        logger.info("Generating summary statistics")
        
        stats = {
            "Analysis Runtime": f"{time.time() - self.start_time:.2f} seconds",
            "Data Period": f"{df.index.min()} to {df.index.max()}",
            "Total Trading Days": len(df),
            "Price Statistics": {
                "Starting Price": f"${df['Close'].iloc[0]:.2f}",
                "Ending Price": f"${df['Close'].iloc[-1]:.2f}",
                "Highest Price": f"${df['High'].max():.2f}",
                "Lowest Price": f"${df['Low'].min():.2f}",
                "Average Price": f"${df['Close'].mean():.2f}",
                "Price Change": f"{((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100):.2f}%"
            },
            "Technical Indicators": {
                "RSI Range": f"{df['rsi'].min():.2f} to {df['rsi'].max():.2f}",
                "Current MACD": f"{df['macd'].iloc[-1]:.2f}",
                "Average ATR": f"{df['atr'].mean():.2f}",
                "Current MFI": f"{df['mfi'].iloc[-1]:.2f}"
            },
            "Volume Statistics": {
                "Average Volume": f"{df['Volume'].mean():.2f}",
                "Max Volume": f"{df['Volume'].max():.2f}",
                "Current Volume": f"{df['Volume'].iloc[-1]:.2f}"
            }
        }
        
        logger.info("\n=== Summary Statistics ===")
        for category, values in stats.items():
            if isinstance(values, dict):
                logger.info(f"\n{category}:")
                for key, value in values.items():
                    logger.info(f"{key}: {value}")
            else:
                logger.info(f"{category}: {values}")
                
        return stats

    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save results to CSV with error handling"""
        logger.info(f"Saving results to: {output_path}")
        try:
            df.to_csv(output_path, index=False)
            logger.info("Results saved successfully")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def display_wide_chart(df: pd.DataFrame, price_cols: List[str], indicator_cols: List[str]):
    """Display split view of price data and indicators"""
    cprint("\n=== Price Data ===", "white", "on_blue", attrs=['bold'])
    print("\nFirst 10 rows:")
    print(df[price_cols].head(10).to_string(index=True, justify='right'))
    print("\nLast 10 rows:")
    print(df[price_cols].tail(10).to_string(index=True, justify='right'))
    
    cprint("\n=== TA-Lib Technical Indicators ===", "white", "on_blue", attrs=['bold'])
    print("\nFirst 10 rows:")
    print(df[indicator_cols].head(10).to_string(index=True, justify='right'))
    print("\nLast 10 rows:")
    print(df[indicator_cols].tail(10).to_string(index=True, justify='right'))

def main():
    try:
        cprint("\n=== Starting TA-Lib Technical Analysis ===", "white", "on_blue", attrs=['bold'])
        
        # Initialize analysis
        ta_analyzer = TechnicalAnalysisTA()
        ta_analyzer.setup_environment()
        
        # Define file paths
        input_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD.csv'
        output_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_with_ta_lib_indicators.csv'
        
        # Process data
        df = ta_analyzer.load_and_clean_data(input_file)
        df = ta_analyzer.calculate_indicators(df)
        
        # Define columns for display
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Ensure 'Adj Close' exists if needed, else remove
        if 'Adj Close' in df.columns:
            price_cols.append('Adj Close')
        indicator_cols = [
            'sma', 'ema', 'tema', 'wma', 'kama', 'trima',  # TA-Lib Trend
            'rsi', 'macd', 'macdsignal', 'macdhist', 'mom', 'roc', 'willr',  # TA-Lib Momentum
            'atr', 'upperband', 'middleband', 'lowerband',  # TA-Lib Volatility
            'obv', 'mfi'  # TA-Lib Volume
        ]
        
        # Display wide chart
        display_wide_chart(df, price_cols, indicator_cols)
        
        # Generate summary statistics
        ta_analyzer.generate_summary_statistics(df)
        
        # Save results
        ta_analyzer.save_results(df, output_file)
        
        cprint("\n=== TA-Lib Technical Analysis Complete ===", "white", "on_blue", attrs=['bold'])
        cprint(f"Results saved to: {output_file}", "white", "on_blue")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        cprint("An error occurred during the technical analysis. Check the logs for more details.", "red", "on_white", attrs=['bold'])
        sys.exit(1)

if __name__ == "__main__":
    main()
