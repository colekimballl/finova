# pandas_ta_review.py

import pandas as pd
import pandas_ta as ta_pt  # Importing pandas_ta with alias to avoid confusion with talib
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
log_file = log_dir / f'pandas_ta_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

class TechnicalAnalysisPandasTA:
    """Technical Analysis class for calculating and managing indicators using Pandas_TA"""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config path"""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.start_time = time.time()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration for technical indicators"""
        return {
            'pandas_ta_indicators': {  # Pandas_TA indicators
                'stoch_k_length': 14,
                'stoch_d_length': 3,
                'sma_pt_length': 10,
                'ema_pt_length': 10,
                'rsi_pt_length': 14
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
            return TechnicalAnalysisPandasTA._default_config()

    def setup_environment(self) -> None:
        """Initialize and verify environment"""
        logger.info("=== Pandas_TA Technical Analysis Process Starting ===")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Pandas Version: {pd.__version__}")
        try:
            logger.info(f"Pandas_TA Version: {ta_pt.__version__}")  # Log Pandas_TA version
        except AttributeError:
            logger.warning("Pandas_TA version information not available.")
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
        """Calculate all technical indicators using Pandas_TA"""
        logger.info("Starting technical indicator calculations using Pandas_TA")
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Calculate Pandas_TA Indicators
                df = self._calculate_pandas_ta_indicators(df)
                
                # Log any warnings captured during indicator calculations
                if len(w) > 0:
                    for warning in w:
                        logger.warning(f"Calculation warning: {warning.message}")
                        
            logger.info("Pandas_TA technical indicator calculations completed successfully.")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _calculate_pandas_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using Pandas_TA"""
        logger.info("Calculating Pandas_TA technical indicators...")
        c_pt = self.config['pandas_ta_indicators']
        
        # SMA using Pandas_TA
        sma_pt_length = c_pt["sma_pt_length"]
        sma_pt_col = f'sma_pt_{sma_pt_length}'
        df[sma_pt_col] = ta_pt.sma(df['Close'], length=sma_pt_length)
        
        # EMA using Pandas_TA
        ema_pt_length = c_pt["ema_pt_length"]
        ema_pt_col = f'ema_pt_{ema_pt_length}'
        df[ema_pt_col] = ta_pt.ema(df['Close'], length=ema_pt_length)
        
        # RSI using Pandas_TA
        rsi_pt_length = c_pt["rsi_pt_length"]
        rsi_pt_col = f'rsi_pt_{rsi_pt_length}'
        df[rsi_pt_col] = ta_pt.rsi(df['Close'], length=rsi_pt_length)
        
        # Stochastic Oscillator using Pandas_TA
        stoch_k_length = c_pt['stoch_k_length']
        stoch_d_length = c_pt['stoch_d_length']
        stoch = ta_pt.stoch(
            df['High'], 
            df['Low'], 
            df['Close'], 
            k=stoch_k_length, 
            d=stoch_d_length
        )
        # Adjust column names based on Pandas_TA output
        stoch_k_col = f"STOCHk_{stoch_k_length}_{stoch_d_length}_3"
        stoch_d_col = f"STOCHd_{stoch_k_length}_{stoch_d_length}_3"
        if stoch_k_col in stoch.columns and stoch_d_col in stoch.columns:
            df['stoch_k'] = stoch[stoch_k_col]
            df['stoch_d'] = stoch[stoch_d_col]
        else:
            logger.warning("Stochastic Oscillator columns not found in Pandas_TA output.")
        
        # Additional Pandas_TA Indicators can be added here following the same pattern
        
        logger.info("Pandas_TA technical indicators calculated successfully.")
        return df

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
                "Current Pandas_TA RSI": f"{df['rsi_pt_14'].iloc[-1]:.2f}" if 'rsi_pt_14' in df.columns else "N/A"
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

def display_wide_chart(df: pd.DataFrame, price_cols: List[str], pandas_ta_cols: List[str]):
    """Display split view of price data and Pandas_TA indicators"""
    cprint("\n=== Price Data ===", "white", "on_blue", attrs=['bold'])
    print("\nFirst 10 rows:")
    print(df[price_cols].head(10).to_string(index=True, justify='right'))
    print("\nLast 10 rows:")
    print(df[price_cols].tail(10).to_string(index=True, justify='right'))
    
    cprint("\n=== Pandas_TA Technical Indicators ===", "white", "on_blue", attrs=['bold'])
    print("\nFirst 10 rows:")
    print(df[pandas_ta_cols].head(10).to_string(index=True, justify='right'))
    print("\nLast 10 rows:")
    print(df[pandas_ta_cols].tail(10).to_string(index=True, justify='right'))

def main():
    try:
        cprint("\n=== Starting Pandas_TA Technical Analysis ===", "white", "on_blue", attrs=['bold'])
        
        # Initialize analysis
        ta_analyzer = TechnicalAnalysisPandasTA()
        ta_analyzer.setup_environment()
        
        # Define file paths
        input_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD.csv'
        output_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_with_pandas_ta_indicators.csv'
        
        # Process data
        df = ta_analyzer.load_and_clean_data(input_file)
        df = ta_analyzer.calculate_indicators(df)
        
        # Define columns for display
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Ensure 'Adj Close' exists if needed, else remove
        if 'Adj Close' in df.columns:
            price_cols.append('Adj Close')
        pandas_ta_cols = [
            'sma_pt_10', 'ema_pt_10', 'rsi_pt_14',  # Pandas_TA SMA, EMA, RSI
            'stoch_k', 'stoch_d'  # Pandas_TA Stochastic Oscillator
        ]
        
        # Display wide chart
        display_wide_chart(df, price_cols, pandas_ta_cols)
        
        # Generate summary statistics
        ta_analyzer.generate_summary_statistics(df)
        
        # Save results
        ta_analyzer.save_results(df, output_file)
        
        cprint("\n=== Pandas_TA Technical Analysis Complete ===", "white", "on_blue", attrs=['bold'])
        cprint(f"Results saved to: {output_file}", "white", "on_blue")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        cprint("An error occurred during the technical analysis. Check the logs for more details.", "red", "on_white", attrs=['bold'])
        sys.exit(1)

if __name__ == "__main__":
    main()
