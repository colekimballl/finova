import pandas as pd
import talib as ta
import pandas_ta as ta_pt  # Importing pandas_ta with alias to avoid confusion with talib
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
import warnings
import yaml
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from termcolor import cprint
import numpy as np

# Set pandas display options for better readability in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# Setup logging with both file and console output
log_dir = Path('/Users/colekimball/ztech/finova/logs')
log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
log_file = log_dir / f'technical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
            },
            'pandas_ta_indicators': {  # New section for Pandas_TA indicators
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
            return TechnicalAnalysis._default_config()

    def setup_environment(self) -> None:
        """Initialize and verify environment"""
        logger.info("=== Technical Analysis Process Starting ===")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Pandas Version: {pd.__version__}")
        logger.info(f"TA-Lib Version: {ta.__version__}")
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
        """Calculate all technical indicators using TA-Lib and Pandas_TA"""
        logger.info("Starting technical indicator calculations")
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # TA-Lib Indicators
                self._calculate_ta_lib_indicators(df)
                
                # Pandas_TA Indicators
                self._calculate_pandas_ta_indicators(df)
                
                # Log any warnings captured during indicator calculations
                if len(w) > 0:
                    for warning in w:
                        logger.warning(f"Calculation warning: {warning.message}")
                        
            logger.info("Technical indicator calculations completed successfully.")
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
        
    def _calculate_pandas_ta_indicators(self, df: pd.DataFrame) -> None:
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
        df[['stoch_k', 'stoch_d']] = stoch[['STOCHk_14_3_3', 'STOCHd_14_3_3']]
        
        # Additional Pandas_TA Indicators can be added here following the same pattern

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate and log summary statistics"""
        logger.info("Generating summary statistics")
        
        # Extract Pandas_TA RSI value safely
        rsi_pt_length = self.config['pandas_ta_indicators']['rsi_pt_length']
        rsi_pt_col = f"rsi_pt_{rsi_pt_length}"
        if rsi_pt_col in df.columns:
            current_pandas_ta_rsi = df[rsi_pt_col].iloc[-1]
        else:
            current_pandas_ta_rsi = np.nan
            logger.warning(f"Pandas_TA RSI column '{rsi_pt_col}' not found in DataFrame.")
        
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
                "Current MFI": f"{df['mfi'].iloc[-1]:.2f}",
                "Current Pandas_TA RSI": f"{current_pandas_ta_rsi:.2f}" if not pd.isna(current_pandas_ta_rsi) else "N/A"
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

def display_wide_chart(df: pd.DataFrame, price_cols: List[str], indicator_cols: List[str], pandas_ta_cols: List[str]):
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
    
    cprint("\n=== Pandas_TA Technical Indicators ===", "white", "on_blue", attrs=['bold'])
    print("\nFirst 10 rows:")
    print(df[pandas_ta_cols].head(10).to_string(index=True, justify='right'))
    print("\nLast 10 rows:")
    print(df[pandas_ta_cols].tail(10).to_string(index=True, justify='right'))

def main():
    try:
        cprint("\n=== Starting Technical Analysis ===", "white", "on_blue", attrs=['bold'])
        
        # Initialize analysis
        ta_analyzer = TechnicalAnalysis()
        ta_analyzer.setup_environment()
        
        # Define file paths
        input_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD.csv'
        output_file = '/Users/colekimball/ztech/finova/data/historical/BTC-USD_with_indicators.csv'
        
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
        # Define Pandas_TA indicator columns
        pandas_ta_cols = [
            'sma_pt_10', 'ema_pt_10', 'rsi_pt_14',  # Pandas_TA SMA, EMA, RSI
            'stoch_k', 'stoch_d'  # Pandas_TA Stochastic Oscillator
        ]
        
        # Display wide chart
        display_wide_chart(df, price_cols, indicator_cols, pandas_ta_cols)

        # Market Analysis
        cprint("\n=== Current Market Analysis ===", "white", "on_blue", attrs=['bold'])
        
        # Price Action
        current_price = df['Close'].iloc[-1]
        if len(df) >= 2:
            price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100)
        else:
            price_change = 0
            logger.warning("Not enough data points to calculate 24h Change.")
        cprint("\nPrice Action:", "cyan", "on_blue", attrs=['bold'])
        print(f"Current Price: ${current_price:.2f}")
        cprint(f"24h Change: {price_change:.2f}%", "green" if price_change > 0 else "red", "on_blue")
        
        # Trend Analysis
        cprint("\nTrend Analysis:", "cyan", "on_blue", attrs=['bold'])
        sma = df['sma'].iloc[-1]
        ema = df['ema'].iloc[-1]
        atr = df['atr'].iloc[-1]
        trend_strength = "STRONG" if abs(sma - ema) > atr else "WEAK"
        cprint(f"Trend Strength: {trend_strength}", "green" if trend_strength == "STRONG" else "yellow", "on_blue")
        
        # Support/Resistance
        cprint("\nSupport/Resistance Levels:", "cyan", "on_blue", attrs=['bold'])
        recent_highs = df['High'].tail(20)
        recent_lows = df['Low'].tail(20)
        resistance = recent_highs.max()
        support = recent_lows.min()
        cprint(f"Next Resistance: ${resistance:.2f}", "red", "on_blue")
        cprint(f"Next Support: ${support:.2f}", "green", "on_blue")
        
        # Pattern Detection
        cprint("\nPattern Detection:", "cyan", "on_blue", attrs=['bold'])
        patterns = []
        
        # Double Top/Bottom
        highs = df['High'].tail(20)
        lows = df['Low'].tail(20)
        if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
            atr = df['atr'].iloc[-1]
        else:
            atr = 0
            logger.warning("ATR not available or NaN.")
        
        if atr > 0:
            if any(abs(highs - highs.max()) < atr):
                patterns.append(("Double Top Potential", "red"))
            if any(abs(lows - lows.min()) < atr):
                patterns.append(("Double Bottom Potential", "green"))
        
        # Breakouts
        if current_price > resistance:
            patterns.append(("Bullish Breakout", "green"))
        elif current_price < support:
            patterns.append(("Bearish Breakdown", "red"))
            
        # Moving Average Crosses
        if len(df) >= 2:
            if df['sma'].iloc[-2] < df['ema'].iloc[-2] and df['sma'].iloc[-1] > df['ema'].iloc[-1]:
                patterns.append(("Bullish MA Crossover", "green"))
            elif df['sma'].iloc[-2] > df['ema'].iloc[-2] and df['sma'].iloc[-1] < df['ema'].iloc[-1]:
                patterns.append(("Bearish MA Crossover", "red"))
        else:
            logger.warning("Not enough data points to calculate Moving Average Crossovers.")
    
        # RSI Divergence
        if len(df) >= 5:
            price_change_div = df['Close'].iloc[-1] - df['Close'].iloc[-5]
            rsi_change = df['rsi'].iloc[-1] - df['rsi'].iloc[-5]
            if price_change_div > 0 and rsi_change < 0:
                patterns.append(("Bearish RSI Divergence", "red"))
            elif price_change_div < 0 and rsi_change > 0:
                patterns.append(("Bullish RSI Divergence", "green"))
        else:
            logger.warning("Not enough data points to calculate RSI Divergence.")
        
        # Print detected patterns
        if patterns:
            for pattern, color in patterns:
                cprint(f"► {pattern}", color, "on_blue")
        else:
            cprint("No significant patterns detected", "yellow", "on_blue")
        
        # Momentum Analysis
        cprint("\nMomentum Analysis:", "cyan", "on_blue", attrs=['bold'])
        if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
            rsi = df['rsi'].iloc[-1]
        else:
            rsi = np.nan
            logger.warning("RSI not available or NaN.")
        
        if 'macd' in df.columns and 'macdsignal' in df.columns and not pd.isna(df['macd'].iloc[-1]) and not pd.isna(df['macdsignal'].iloc[-1]):
            macd = df['macd'].iloc[-1]
            macd_sig = df['macdsignal'].iloc[-1]
        else:
            macd = np.nan
            macd_sig = np.nan
            logger.warning("MACD or MACD Signal not available or NaN.")
        
        if not pd.isna(rsi):
            cprint(f"RSI: {rsi:.2f}", "green" if rsi > 50 else "red", "on_blue")
        else:
            cprint("RSI: N/A", "red", "on_blue")
        
        if not pd.isna(macd) and not pd.isna(macd_sig):
            cprint(f"MACD: {macd:.2f}", "green" if macd > macd_sig else "red", "on_blue")
        else:
            cprint("MACD: N/A", "red", "on_blue")
        
        # Volume Analysis
        cprint("\nVolume Analysis:", "cyan", "on_blue", attrs=['bold'])
        if 'Volume' in df.columns and not pd.isna(df['Volume'].iloc[-1]):
            vol_avg = df['Volume'].tail(20).mean()
            curr_vol = df['Volume'].iloc[-1]
            vol_status = "HIGH" if curr_vol > vol_avg * 1.5 else "LOW" if curr_vol < vol_avg * 0.5 else "NORMAL"
            cprint(f"Volume Status: {vol_status}", "yellow", "on_blue")
        else:
            vol_status = "N/A"
            cprint(f"Volume Status: {vol_status}", "yellow", "on_blue")
        
        # Alerts Section
        cprint("\n=== Trading Alerts ===", "magenta", "on_blue", attrs=['bold'])
        
        alerts = []
        
        # Strong Buy Conditions
        if (not pd.isna(rsi) and rsi < 30) and \
           (not pd.isna(macd) and not pd.isna(macd_sig) and macd > macd_sig) and \
           (current_price > support):
            alerts.append(("STRONG BUY", "green", [
                "RSI oversold condition",
                "MACD bullish crossover",
                "Price above support"
            ]))
            
        # Strong Sell Conditions    
        elif (not pd.isna(rsi) and rsi > 70) and \
             (not pd.isna(macd) and not pd.isna(macd_sig) and macd < macd_sig) and \
             (current_price < resistance):
            alerts.append(("STRONG SELL", "red", [
                "RSI overbought condition",
                "MACD bearish crossover",
                "Price below resistance"
            ]))
            
        # Caution Conditions
        if (vol_status == "HIGH" or (not pd.isna(price_change) and abs(price_change) > 5)):
            caution_reasons = []
            if vol_status == "HIGH":
                caution_reasons.append("Unusual volume activity")
            if not pd.isna(price_change) and abs(price_change) > 5:
                caution_reasons.append(f"Large price movement: {price_change:.2f}%")
            alerts.append(("CAUTION", "yellow", caution_reasons))

        # Print Alerts
        if alerts:
            for signal, color, reasons in alerts:
                cprint(f"\n► {signal} Signal", color, "on_blue", attrs=['bold'])
                for reason in reasons:
                    if reason:  # Only print non-empty reasons
                        cprint(f"  • {reason}", color, "on_blue")
        else:
            cprint("\nNo significant alerts detected at this time", "yellow", "on_blue")
        
        # Generate summary statistics
        ta_analyzer.generate_summary_statistics(df)
        
        # Save results
        ta_analyzer.save_results(df, output_file)
        
        cprint("\n=== Technical Analysis Complete ===", "white", "on_blue", attrs=['bold'])
        cprint(f"Results saved to: {output_file}", "white", "on_blue")

    except: 
        print('oops')
    
    if __name__ == "__main__":
        main()
