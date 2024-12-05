import pandas as pd
import talib as ta
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
import warnings
import yaml
from typing import Dict, Any, Tuple

# Setup logging
log_dir = Path('/Users/colekimball/ztech/finova/logs')
log_file = log_dir / f'technical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Technical Analysis class for calculating and managing indicators"""
    
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
            return TechnicalAnalysis._default_config()

    def setup_environment(self) -> None:
        """Initialize and verify environment"""
        logger.info("=== Technical Analysis Process Starting ===")
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
            
            # Clean data
            df = df.iloc[2:].reset_index(drop=True)
            logger.info("Removed header rows")
            
            # Convert price columns
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
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
        """Calculate all technical indicators"""
        logger.info("Starting technical indicator calculations")
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Trend Indicators
                logger.info("Calculating trend indicators...")
                c = self.config['trend_indicators']
                df['sma'] = ta.SMA(df['Close'], timeperiod=c['sma_period'])
                df['ema'] = ta.EMA(df['Close'], timeperiod=c['ema_period'])
                df['tema'] = ta.TEMA(df['Close'], timeperiod=c['tema_period'])
                df['wma'] = ta.WMA(df['Close'], timeperiod=c['wma_period'])
                df['kama'] = ta.KAMA(df['Close'], timeperiod=c['kama_period'])
                df['trima'] = ta.TRIMA(df['Close'], timeperiod=c['trima_period'])
                
                # Momentum Indicators
                logger.info("Calculating momentum indicators...")
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
                
                # Volatility Indicators
                logger.info("Calculating volatility indicators...")
                c = self.config['volatility_indicators']
                df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=c['atr_period'])
                df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(
                    df['Close'],
                    timeperiod=c['bbands_period'],
                    nbdevup=c['bbands_dev'],
                    nbdevdn=c['bbands_dev']
                )
                
                # Volume Indicators
                logger.info("Calculating volume indicators...")
                c = self.config['volume_indicators']
                df['obv'] = ta.OBV(df['Close'], df['Volume'])
                df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=c['mfi_period'])
                
                if len(w) > 0:
                    for warning in w:
                        logger.warning(f"Calculation warning: {warning.message}")
                        
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate and log summary statistics"""
        logger.info("Generating summary statistics")
        
        stats = {
            "Analysis Runtime": f"{time.time() - self.start_time:.2f} seconds",
            "Data Period": f"{df.index[0]} to {df.index[-1]}",
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

def main():
    try:
        # Initialize analysis
        ta_analyzer = TechnicalAnalysis()
        ta_analyzer.setup_environment()
        
        # Define file paths
        input_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD.csv'
        output_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_with_indicators.csv'
        
        # Process data
        df = ta_analyzer.load_and_clean_data(input_file)
        df = ta_analyzer.calculate_indicators(df)
        stats = ta_analyzer.generate_summary_statistics(df)
        ta_analyzer.save_results(df, output_file)
        
        logger.info("=== Technical Analysis Process Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()