# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import logging
from pathlib import Path
import traceback
import warnings
import requests
import threading
import time
import os

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Ensure Streamlit version is up-to-date
if float('.'.join(st.__version__.split('.')[:2])) < 1.18:
    st.error("Please upgrade Streamlit to version 1.18.0 or higher.")
    st.stop()

# Check if TA-Lib is installed
try:
    import talib as ta
    TALIB_INSTALLED = True
except ImportError:
    TALIB_INSTALLED = False
    st.error("TA-Lib not installed. Please install it using: `conda install -c conda-forge ta-lib` or `pip install ta-lib`")
    st.stop()

# List of symbols to track (yfinance compatible)
TRACK_SYMBOLS = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "DOGE-USD": "DOGE-USD",
    "ADA-USD": "ADA-USD",
    "SOL-USD": "SOL-USD"
}

# Initialize logging
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

# Placeholder Strategy Classes
class SmaCross:
    pass

class SmaRsiStrategy:
    pass

class TradingDashboard:
    def __init__(self):
        st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide")
        
        # Setup logging
        script_dir = Path(__file__).parent.parent  # Adjust if necessary
        log_dir = script_dir / 'scripts' / 'application' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("TradingDashboard", log_dir / "dashboard.log")
        
        # Initialize session state
        self.initialize_session_state()
        
        # Initialize indicator configurations
        self.setup_indicators()
        
        # Initialize live data storage
        self.live_data = {
            'big_liqs': [],
            # Add other data streams as needed
        }
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Start threads to read CSV files
        threading.Thread(target=self.read_big_liqs_data, daemon=True).start()
        # Add similar threads for other CSVs if needed
        
        self.logger.info("Trading Dashboard initialized")
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        default_values = {
            'selected_symbol': 'BTC-USD',
            'timeframe': '1d',
            'backtesting_results': None,
            'max_position': 10000.0,
            # Indicator parameters
            'sma_period': 20,
            'ema_period': 20,
            'tema_period': 20,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'bbands_period': 20,
            'bbands_dev': 2.0,
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def setup_indicators(self):
        """Initialize technical indicators configuration"""
        self.indicator_config = {
            "trend_indicators": {
                "sma_period": st.session_state.sma_period,
                "ema_period": st.session_state.ema_period,
                "tema_period": st.session_state.tema_period,
            },
            "momentum_indicators": {
                "rsi_period": st.session_state.rsi_period,
                "macd_fast": st.session_state.macd_fast,
                "macd_slow": st.session_state.macd_slow,
                "macd_signal": st.session_state.macd_signal,
            },
            "volatility_indicators": {
                "atr_period": st.session_state.atr_period,
                "bbands_period": st.session_state.bbands_period,
                "bbands_dev": st.session_state.bbands_dev,
            }
        }
    
    def safe_calculate_indicator(self, func, *args, **kwargs):
        """Safely calculate an indicator with error handling"""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Error calculating indicator {func.__name__}: {str(e)}")
            return np.full(len(args[0]), np.nan)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with proper error handling"""
        if not TALIB_INSTALLED:
            self.logger.error("TA-Lib not installed. Skipping indicator calculations.")
            return df

        try:
            # Convert to numpy arrays for TA-Lib
            close_arr = df['Close'].values  # 1D array
            high_arr = df['High'].values
            low_arr = df['Low'].values
            
            with st.spinner('Calculating technical indicators...'):
                # Trend indicators
                df['SMA'] = self.safe_calculate_indicator(
                    ta.SMA, close_arr, timeperiod=int(self.indicator_config['trend_indicators']['sma_period']))
                df['EMA'] = self.safe_calculate_indicator(
                    ta.EMA, close_arr, timeperiod=int(self.indicator_config['trend_indicators']['ema_period']))
                
                # Momentum indicators
                df['RSI'] = self.safe_calculate_indicator(
                    ta.RSI, close_arr, timeperiod=int(self.indicator_config['momentum_indicators']['rsi_period']))
                
                # MACD
                macd_res = self.safe_calculate_indicator(
                    ta.MACD, close_arr,
                    fastperiod=int(self.indicator_config['momentum_indicators']['macd_fast']),
                    slowperiod=int(self.indicator_config['momentum_indicators']['macd_slow']),
                    signalperiod=int(self.indicator_config['momentum_indicators']['macd_signal']))
                
                if isinstance(macd_res, tuple) and len(macd_res) == 3:
                    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd_res
                
                # Volatility indicators
                df['ATR'] = self.safe_calculate_indicator(
                    ta.ATR, high_arr, low_arr, close_arr,
                    timeperiod=int(self.indicator_config['volatility_indicators']['atr_period']))
                
                # Bollinger Bands
                bb_res = self.safe_calculate_indicator(
                    ta.BBANDS, close_arr,
                    timeperiod=int(self.indicator_config['volatility_indicators']['bbands_period']),
                    nbdevup=float(self.indicator_config['volatility_indicators']['bbands_dev']),
                    nbdevdn=float(self.indicator_config['volatility_indicators']['bbands_dev']))
                
                if isinstance(bb_res, tuple) and len(bb_res) == 3:
                    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = bb_res
            
            return df
        except Exception as e:
            self.logger.error(f"Error in calculate_indicators: {str(e)}\n{traceback.format_exc()}")
            st.error("Error calculating indicators. Check logs for details.")
            return df
    
    def fetch_data(self, symbol: str, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
        """Fetch market data safely with error handling and loading state"""
        try:
            with st.spinner(f'Fetching {symbol} data...'):
                data = yf.download(symbol, period=period, interval=interval)
                
                if data.empty:
                    st.warning("No data fetched. Please check the symbol and try again.")
                    self.logger.error(f"No data fetched for {symbol}")
                    return pd.DataFrame()
                
                if len(data) < 2:
                    st.warning("Insufficient data points for the selected timeframe")
                    self.logger.error(f"Insufficient data points for {symbol} with period={period}, interval={interval}")
                    return pd.DataFrame()
                
                # Calculate indicators
                data = self.calculate_indicators(data)
                return data
                
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}\n{traceback.format_exc()}")
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def read_big_liqs_data(self):
        """Continuously read binance_bigliqs.csv and update live_data"""
        while True:
            try:
                # Construct the path relative to the script's directory
                script_dir = Path(__file__).parent.parent  # Adjust if necessary
                csv_path = script_dir / 'data' / 'raw' / 'binance_bigliqs.csv'
                
                if not csv_path.exists():
                    self.logger.error(f"Error reading binance_bigliqs.csv: [Errno 2] No such file or directory: '{csv_path}'")
                    time.sleep(5)
                    continue  # Retry after delay
                
                # Read the latest data
                df = pd.read_csv(csv_path)
                
                # Validate expected columns
                expected_columns = {'timestamp', 'price'}  # Adjust based on your CSV structure
                if not expected_columns.issubset(df.columns):
                    self.logger.error(f"CSV file is missing required columns. Found columns: {df.columns.tolist()}")
                    time.sleep(5)
                    continue
                
                if not df.empty:
                    latest_entry = df.iloc[-1]
                    with self.lock:
                        self.live_data['big_liqs'].append({
                            'timestamp': latest_entry.get('timestamp', datetime.now().isoformat()),
                            'price': latest_entry.get('price', np.nan)
                        })
                        # Keep only the latest 100 entries
                        if len(self.live_data['big_liqs']) > 100:
                            self.live_data['big_liqs'].pop(0)
                
                time.sleep(5)  # Adjust the sleep time as needed
            
            except Exception as e:
                self.logger.error(f"Error reading binance_bigliqs.csv: {e}\n{traceback.format_exc()}")
                time.sleep(5)
    
    def create_price_chart(self, data: pd.DataFrame, ticker: str) -> None:
        """Create interactive price chart with error handling"""
        try:
            if data.empty:
                st.warning("No data available to display price chart.")
                return
            
            with st.spinner(f'Creating price chart for {ticker}...'):
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                # Add indicators if available
                if 'SMA' in data.columns and not data['SMA'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['SMA'],
                        name=f'SMA ({int(self.indicator_config["trend_indicators"]["sma_period"])})',
                        line=dict(color='blue')
                    ))
                
                if 'EMA' in data.columns and not data['EMA'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['EMA'],
                        name=f'EMA ({int(self.indicator_config["trend_indicators"]["ema_period"])})',
                        line=dict(color='orange')
                    ))
                
                if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']) and not data['BB_Upper'].isna().all():
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', dash='dash'),
                        fill='tonexty'
                    ))
                
                fig.update_layout(
                    title=f'{ticker} Price Chart',
                    yaxis_title='Price (USD)',
                    height=600,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            self.logger.error(f"Error creating price chart for {ticker}: {str(e)}\n{traceback.format_exc()}")
            st.error("Error creating price chart. Check logs for details.")
    
    def display_metrics(self, data: pd.DataFrame) -> None:
        """Display key trading metrics with error handling"""
        try:
            if data.empty:
                st.warning("No data available to display metrics.")
                return
            
            with st.spinner('Calculating metrics...'):
                metrics = {}
                
                # Basic metrics
                metrics['Last Price'] = float(data['Close'].iloc[-1])
                metrics['Daily Return'] = float(((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100))
                metrics['Volatility (Ann.)'] = float(data['Close'].pct_change().std() * np.sqrt(252) * 100)
                
                # Technical indicators
                if 'RSI' in data.columns and not data['RSI'].isna().all():
                    metrics['RSI'] = float(data['RSI'].iloc[-1])
                else:
                    metrics['RSI'] = np.nan
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Last Price", f"${metrics['Last Price']:,.2f}")
                with col2:
                    st.metric("Daily Return", f"{metrics['Daily Return']:,.2f}%",
                              delta=f"{metrics['Daily Return']:,.2f}%")
                with col3:
                    if not np.isnan(metrics['RSI']):
                        st.metric("RSI", f"{metrics['RSI']:,.2f}")
                    else:
                        st.metric("RSI", "N/A")
                with col4:
                    st.metric("Volatility (Ann.)", f"{metrics['Volatility (Ann.)']:,.2f}%")
                    
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}\n{traceback.format_exc()}")
            st.error("Error calculating metrics. Check logs for details.")
    
    def technical_indicators_chart(self, data: pd.DataFrame) -> None:
        """Display technical indicators charts with error handling"""
        try:
            if data.empty:
                st.warning("No data available to display technical indicators.")
                return
            
            with st.spinner('Creating indicator charts...'):
                # RSI Chart
                if 'RSI' in data.columns and not data['RSI'].isna().all():
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title='RSI', height=250, template='plotly_dark')
                
                # MACD Chart
                macd_available = all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']) and not data[['MACD', 'MACD_Signal', 'MACD_Hist']].isna().all().all()
                if macd_available:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='orange')))
                    fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color='grey'))
                    fig_macd.update_layout(title='MACD', height=250, template='plotly_dark')
                
                # Display charts
                col1, col2 = st.columns(2)
                with col1:
                    if 'RSI' in data.columns and not data['RSI'].isna().all():
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    else:
                        st.warning("RSI data not available")
                with col2:
                    if macd_available:
                        st.plotly_chart(fig_macd, use_container_width=True)
                    else:
                        st.warning("MACD data not available")
                        
        except Exception as e:
            self.logger.error(f"Error creating indicator charts: {str(e)}\n{traceback.format_exc()}")
            st.error("Error creating indicator charts. Check logs for details.")
    
    def test_connections_page(self):
        """Render API connection testing page"""
        st.header("🔧 API Connection Testing")
        st.write("Test connections to Phemex, Hyperliquid, and Coinbase APIs.")
        
        # Initialize button states
        if 'phemex_tested' not in st.session_state:
            st.session_state.phemex_tested = False
        if 'hyperliquid_tested' not in st.session_state:
            st.session_state.hyperliquid_tested = False
        if 'coinbase_tested' not in st.session_state:
            st.session_state.coinbase_tested = False
        
        # Placeholder functions for testing API connections
        def test_phemex_connection():
            # TODO: Implement actual API connection test
            try:
                # Simulate API test
                time.sleep(1)
                st.session_state.phemex_tested = True
                st.success("Phemex API connection successful!")
            except Exception as e:
                st.error(f"Phemex API connection failed: {e}")
        
        def test_hyperliquid_connection():
            # TODO: Implement actual API connection test
            try:
                # Simulate API test
                time.sleep(1)
                st.session_state.hyperliquid_tested = True
                st.success("Hyperliquid API connection successful!")
            except Exception as e:
                st.error(f"Hyperliquid API connection failed: {e}")
        
        def test_coinbase_connection():
            # TODO: Implement actual API connection test
            try:
                # Simulate API test
                time.sleep(1)
                st.session_state.coinbase_tested = True
                st.success("Coinbase API connection successful!")
            except Exception as e:
                st.error(f"Coinbase API connection failed: {e}")
        
        # Create buttons for each API test
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Test Phemex Connection"):
                test_phemex_connection()
        
        with col2:
            if st.button("Test Hyperliquid Connection"):
                test_hyperliquid_connection()
        
        with col3:
            if st.button("Test Coinbase Connection"):
                test_coinbase_connection()
        
        st.write("**Note:** These are placeholder tests. Implement actual API connection logic as needed.")
    
    def backtesting_playground_page(self):
        """Render backtesting playground page"""
        st.header("📊 Backtesting Playground")
        st.write("Visualize and interact with your backtest results.")
        
        # Placeholder for backtesting visualization
        # TODO: Integrate actual backtesting data and visualization
        if st.button("Load Backtest Results"):
            try:
                # Replace with actual path and loading mechanism
                script_dir = Path(__file__).parent.parent
                backtest_path = script_dir / 'backtesting' / 'backtest_results.csv'
                
                if not backtest_path.exists():
                    st.error("Backtest results file not found. Please run a backtest first.")
                    self.logger.error("Backtest results file not found.")
                    return
                
                backtest_results = pd.read_csv(backtest_path)
                st.write(backtest_results)
                
                # Example plot
                if 'returns' in backtest_results.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=backtest_results['returns'], mode='lines', name='Returns'))
                    fig.update_layout(title='Backtest Returns', xaxis_title='Time', yaxis_title='Cumulative Returns', template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No 'returns' column found in backtest results.")
            except Exception as e:
                self.logger.error(f"Error loading backtest results: {e}")
                st.error(f"Error loading backtest results: {e}")
        
        st.write("**Note:** Integrate your backtesting framework to generate and visualize actual results.")
    
    def run_backtest(self, data: pd.DataFrame) -> dict:
        """Run backtest using selected strategy"""
        try:
            with st.spinner('Running backtest...'):
                # Convert data to format expected by backtesting
                if 'SMA' not in data.columns or 'RSI' not in data.columns:
                    st.warning("Required indicators not available for backtesting")
                    return {}
                
                # Select strategy based on user choice
                # Placeholder: Replace with actual strategy selection if needed
                strategy_class = SmaCross
                strategy_params = {'sma_period': st.session_state.sma_period}
                
                # Placeholder for actual backtest logic
                # Replace this with your backtesting framework integration
                results = {
                    'strategy': 'SMA Crossover',
                    'params': strategy_params,
                    'returns': np.random.randn(100).cumsum().tolist(),  # Dummy data
                    'metrics': {
                        'Total Return': f"{np.random.uniform(5, 20):.2f}%",
                        'Max Drawdown': f"{np.random.uniform(-10, -2):.2f}%",
                        'Sharpe Ratio': f"{np.random.uniform(1, 3):.2f}"
                    }
                }
                
                self.logger.info(f"Backtest completed: {results}")
                return results
                
        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}\n{traceback.format_exc()}")
            st.error("Error running backtest. Check logs for details.")
            return {}
    
    def display_backtest_results(self, results: dict):
        """Display backtest results"""
        try:
            if not results:
                st.warning("No backtest results to display.")
                return
            
            st.subheader("Backtest Results")
            st.write(f"**Strategy:** {results.get('strategy', 'N/A')}")
            st.write(f"**Parameters:** {results.get('params', {})}")
            
            metrics = results.get('metrics', {})
            if metrics:
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                st.table(metrics_df)
            
            # Plot returns
            if 'returns' in results and results['returns']:
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Scatter(y=results['returns'], mode='lines', name='Cumulative Returns'))
                fig_returns.update_layout(title='Backtest Cumulative Returns', template='plotly_dark')
                st.plotly_chart(fig_returns, use_container_width=True)
                
        except Exception as e:
            self.logger.error(f"Error displaying backtest results: {e}\n{traceback.format_exc()}")
            st.error("Error displaying backtest results. Check logs for details.")
    
    def run(self):
        """Main method to run the dashboard"""
        st.title("📈 Algorithmic Trading Dashboard")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            app_mode = st.radio("Go to", ["Data Dashboard", "API Connection Testing", "Backtesting Playground"])
        
        # Render the selected page
        if app_mode == "Data Dashboard":
            st.subheader("📊 Market Data")
            # Sidebar inputs
            with st.sidebar:
                st.header("Configuration")
                st.session_state.selected_symbol = st.selectbox(
                    "Symbol",
                    options=list(TRACK_SYMBOLS.values()),
                    index=list(TRACK_SYMBOLS.values()).index(st.session_state.selected_symbol)
                )
                st.session_state.timeframe = st.selectbox(
                    "Timeframe",
                    options=["1d", "1wk", "1mo"],
                    index=["1d", "1wk", "1mo"].index(st.session_state.timeframe)
                )
                # Technical Indicators Configuration
                with st.expander("Technical Indicators Configuration"):
                    # SMA Period
                    st.session_state.sma_period = st.number_input(
                        "SMA Period", min_value=5, max_value=200, value=int(st.session_state.sma_period), step=1
                    )
                    # EMA Period
                    st.session_state.ema_period = st.number_input(
                        "EMA Period", min_value=5, max_value=200, value=int(st.session_state.ema_period), step=1
                    )
                    # TEMA Period
                    st.session_state.tema_period = st.number_input(
                        "TEMA Period", min_value=5, max_value=200, value=int(st.session_state.tema_period), step=1
                    )
                    # RSI Period
                    st.session_state.rsi_period = st.number_input(
                        "RSI Period", min_value=5, max_value=50, value=int(st.session_state.rsi_period), step=1
                    )
                    # MACD Fast Period
                    st.session_state.macd_fast = st.number_input(
                        "MACD Fast Period", min_value=5, max_value=50, value=int(st.session_state.macd_fast), step=1
                    )
                    # MACD Slow Period
                    st.session_state.macd_slow = st.number_input(
                        "MACD Slow Period", min_value=10, max_value=100, value=int(st.session_state.macd_slow), step=1
                    )
                    # MACD Signal Period
                    st.session_state.macd_signal = st.number_input(
                        "MACD Signal Period", min_value=5, max_value=50, value=int(st.session_state.macd_signal), step=1
                    )
                    # ATR Period
                    st.session_state.atr_period = st.number_input(
                        "ATR Period", min_value=5, max_value=50, value=int(st.session_state.atr_period), step=1
                    )
                    # Bollinger Bands Period
                    st.session_state.bbands_period = st.number_input(
                        "Bollinger Bands Period", min_value=5, max_value=100, value=int(st.session_state.bbands_period), step=1
                    )
                    # Bollinger Bands Deviation
                    st.session_state.bbands_dev = st.number_input(
                        "Bollinger Bands Deviation", min_value=1.0, max_value=5.0, value=float(st.session_state.bbands_dev), step=0.1
                    )
        
            # Fetch data
            data = self.fetch_data(
                symbol=st.session_state.selected_symbol,
                period=self.get_period_from_timeframe(st.session_state.timeframe),
                interval=self.get_interval_from_timeframe(st.session_state.timeframe)
            )
            
            if not data.empty:
                # Display metrics
                self.display_metrics(data)
                
                # Price chart
                self.create_price_chart(data, st.session_state.selected_symbol)
                
                # Technical indicators
                self.technical_indicators_chart(data)
            
            # Live Data Display
            st.subheader("📈 Live Data Streams")
            with st.spinner("Loading live data..."):
                with self.lock:
                    big_liqs = self.live_data.get('big_liqs', [])
                    # Add other data streams as needed
    
                if big_liqs:
                    live_df = pd.DataFrame(big_liqs)
                    # Display as a table
                    st.table(live_df.tail(10))  # Show last 10 entries
                    
                    # Plotting example
                    if 'price' in live_df.columns and 'timestamp' in live_df.columns:
                        fig_live = go.Figure()
                        fig_live.add_trace(go.Scatter(x=pd.to_datetime(live_df['timestamp']), y=live_df['price'], mode='lines+markers', name='Price'))
                        fig_live.update_layout(
                            title='Live Price Data',
                            xaxis_title='Timestamp',
                            yaxis_title='Price (USD)',
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig_live, use_container_width=True)
                else:
                    st.write("No live data available yet.")
        
        elif app_mode == "API Connection Testing":
            self.test_connections_page()
        
        elif app_mode == "Backtesting Playground":
            self.backtesting_playground_page()
    
    def get_period_from_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to yfinance period"""
        mapping = {
            "1d": "6mo",
            "1wk": "1y",
            "1mo": "5y"
        }
        return mapping.get(timeframe, "6mo")
    
    def get_interval_from_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to yfinance interval"""
        mapping = {
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo"
        }
        return mapping.get(timeframe, "1d")

# Instantiate and run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()

