'''
Project Solaris AI - Technical Analysis Indicators

This module provides a comprehensive set of technical indicators for the
Cardano trading system, leveraging the pandas-ta library for efficient calculations.
'''

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/ta_indicators.log'
)
logger = logging.getLogger("TA-Indicators")

class TechnicalAnalysis:
    def __init__(self, data_dir='data/cardano'):
        """
        Initialize the Technical Analysis class
        
        Parameters:
        -----------
        data_dir: str
            Directory containing the price data
        """
        self.data_dir = data_dir
        logger.info(f"TA module initialized with data directory: {data_dir}")
        
    def load_data(self, timeframe='1h', recent_days=None):
        """
        Load price data for the specified timeframe
        
        Parameters:
        -----------
        timeframe: str
            Timeframe of the data to load ('1m', '5m', '15m', '1h', '4h', '1d')
        recent_days: int
            If specified, only load the most recent N days of data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with price data
        """
        # Find the appropriate data file
        if timeframe == '1m':
            filename = f"ADAUSD-1m-master.csv"
        else:
            filename = f"ADAUSD-{timeframe}.csv"
            
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Filter for recent days if specified
            if recent_days:
                df = df.iloc[-recent_days*24:] if 'h' in timeframe else df.iloc[-recent_days:]
                
            logger.info(f"Loaded {len(df)} rows of {timeframe} data")
            return df
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def add_all_indicators(self, df):
        """
        Add a comprehensive set of technical indicators to the dataframe
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added indicators
        """
        if df is None or df.empty:
            logger.warning("Cannot add indicators to empty dataframe")
            return None
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Moving Averages
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # Volume Weighted Moving Average
        df['vwma_20'] = ta.vwma(df['close'], df['volume'], length=20)
        df['vwma_40'] = ta.vwma(df['close'], df['volume'], length=40)
        
        # Momentum Indicators
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['stoch_k'], df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        
        # ATR for volatility
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        df['tenkan'] = ichimoku['ITS_9']  # Conversion Line
        df['kijun'] = ichimoku['IKS_26']   # Base Line
        
        # Volume Indicators
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend Indicators
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
        
        # Add crossover signals
        self._add_crossover_signals(df)
        
        return df
        
    def _add_crossover_signals(self, df):
        """Add crossover signals to the dataframe"""
        # EMA Crossovers (Golden/Death Cross)
        df['ema_12_26_cross'] = ta.cross(df['ema_12'], df['ema_26'])
        
        # Golden Cross (50 crosses above 200)
        df['golden_cross'] = ta.cross(df['sma_50'], df['sma_200'])
        
        # Death Cross (50 crosses below 200)
        df['death_cross'] = ta.cross(df['sma_200'], df['sma_50'])
        
        # MACD Signal Line Crossover
        df['macd_cross'] = ta.cross(df['macd'], df['macd_signal'])
        
        # RSI crosses
        df['rsi_oversold'] = ((df['rsi_14'] < 30) & (df['rsi_14'].shift(1) >= 30))
        df['rsi_overbought'] = ((df['rsi_14'] > 70) & (df['rsi_14'].shift(1) <= 70))
        
        # Price crosses above/below key MAs
        df['price_cross_sma_20'] = ta.cross(df['close'], df['sma_20'])
        df['price_cross_sma_50'] = ta.cross(df['close'], df['sma_50'])
        
        # VWMA Crossovers
        df['vwma_cross'] = ta.cross(df['vwma_20'], df['vwma_40'])
        
        return df
    
    def get_vwap_analysis(self, df):
        """
        Calculate VWAP and related metrics
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with VWAP analysis
        """
        df = df.copy()
        
        # Calculate VWAP
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Calculate price deviation from VWAP
        df['vwap_deviation'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        
        # VWAP based signals
        df['above_vwap'] = df['close'] > df['vwap']
        df['vwap_cross'] = ta.cross(df['close'], df['vwap'])
        
        return df
    
    def get_vwma_analysis(self, df, lengths=[20, 40, 75]):
        """
        Calculate Volume Weighted Moving Averages for multiple lengths
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price data
        lengths: list
            List of periods to calculate VWMAs for
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with VWMA analysis
        """
        df = df.copy()
        
        # Calculate VWMAs for each length
        for length in lengths:
            df[f'vwma_{length}'] = ta.vwma(df['close'], df['volume'], length=length)
            
            # Add signals based on price vs VWMA
            df[f'above_vwma_{length}'] = df['close'] > df[f'vwma_{length}']
            df[f'vwma_{length}_cross'] = ta.cross(df['close'], df[f'vwma_{length}'])
            
        # Add VWMA crossovers
        if len(lengths) > 1:
            for i in range(len(lengths)-1):
                short = lengths[i]
                long = lengths[i+1]
                df[f'vwma_{short}_{long}_cross'] = ta.cross(
                    df[f'vwma_{short}'], 
                    df[f'vwma_{long}']
                )
        
        return df
    
    def plot_chart(self, df, indicators=None, title=None, save_path=None):
        """
        Plot a price chart with selected indicators
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price and indicator data
        indicators: list
            List of indicators to plot
        title: str
            Chart title
        save_path: str
            Path to save the chart image
        """
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'rsi_14']
            
        # Set up the plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        axes[0].plot(df.index, df['close'], label='Close Price', color='black')
        
        # Plot selected indicators
        for indicator in indicators:
            if indicator in df.columns:
                # Skip boolean columns
                if df[indicator].dtype == bool:
                    continue
                    
                # Plot indicators that should be on the price chart
                if any(x in indicator for x in ['sma', 'ema', 'vwma', 'bb_', 'vwap']):
                    axes[0].plot(df.index, df[indicator], label=indicator)
            
        # Plot RSI in the second subplot
        if 'rsi_14' in df.columns:
            axes[1].plot(df.index, df['rsi_14'], label='RSI (14)', color='purple')
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
            axes[1].set_ylim(0, 100)
            axes[1].set_ylabel('RSI Value')
            axes[1].legend(loc='best')
            
        # Set titles and labels
        if title:
            fig.suptitle(title, fontsize=16)
        axes[0].set_title('Price and Indicators')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Format x-axis
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Chart saved to {save_path}")
            
        return fig, axes
        
    def scan_for_signals(self, df, strategy='all'):
        """
        Scan the most recent data for trading signals
        
        Parameters:
        -----------
        df: pandas.DataFrame
            DataFrame with price and indicator data
        strategy: str
            Strategy to use for scanning ('all', 'ma_cross', 'rsi', 'vwma', etc.)
            
        Returns:
        --------
        dict
            Dictionary with trading signals
        """
        if df is None or df.empty:
            return {'error': 'No data available'}
            
        # Get the most recent candle
        latest = df.iloc[-1]
        
        signals = {
            'timestamp': latest.name,
            'price': latest['close'],
            'signals': {}
        }
        
        # MA Crossovers
        if strategy in ['all', 'ma_cross']:
            signals['signals']['ema_cross'] = bool(latest.get('ema_12_26_cross', False))
            signals['signals']['golden_cross'] = bool(latest.get('golden_cross', False))
            signals['signals']['death_cross'] = bool(latest.get('death_cross', False))
            
        # RSI signals
        if strategy in ['all', 'rsi']:
            signals['signals']['rsi_value'] = latest.get('rsi_14', None)
            signals['signals']['rsi_oversold'] = bool(latest.get('rsi_oversold', False))
            signals['signals']['rsi_overbought'] = bool(latest.get('rsi_overbought', False))
            
        # MACD signals
        if strategy in ['all', 'macd']:
            signals['signals']['macd_cross'] = bool(latest.get('macd_cross', False))
            signals['signals']['macd_positive'] = latest.get('macd', 0) > 0
            signals['signals']['macd_hist_positive'] = latest.get('macd_hist', 0) > 0
            
        # VWMA signals
        if strategy in ['all', 'vwma']:
            signals['signals']['vwma_cross'] = bool(latest.get('vwma_cross', False))
            signals['signals']['above_vwma_20'] = bool(latest.get('above_vwma_20', False))
            
        # Price vs key levels
        if strategy in ['all', 'price_action']:
            signals['signals']['above_sma_20'] = latest['close'] > latest.get('sma_20', float('inf'))
            signals['signals']['above_sma_50'] = latest['close'] > latest.get('sma_50', float('inf'))
            signals['signals']['above_sma_200'] = latest['close'] > latest.get('sma_200', float('inf'))
            
            # Bollinger Bands
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                signals['signals']['near_bb_upper'] = latest['close'] > latest['bb_upper'] * 0.98
                signals['signals']['near_bb_lower'] = latest['close'] < latest['bb_lower'] * 1.02
        
        # Overall signal strength (simple algorithm)
        buy_signals = sum(1 for k, v in signals['signals'].items() 
                         if isinstance(v, bool) and v and 'bear' not in k and 'death' not in k and 'overbought' not in k)
        sell_signals = sum(1 for k, v in signals['signals'].items() 
                          if isinstance(v, bool) and v and ('bear' in k or 'death' in k or 'overbought' in k))
        
        signals['overall'] = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'net_signal': buy_signals - sell_signals,
            'signal_strength': (buy_signals - sell_signals) / max(1, buy_signals + sell_signals)
        }
        
        return signals

# Example usage
if __name__ == "__main__":
    ta = TechnicalAnalysis()
    
    # Load hourly data for the last 7 days
    df = ta.load_data(timeframe='1h', recent_days=7)
    
    if df is not None:
        # Add indicators
        df = ta.add_all_indicators(df)
        
        # Scan for signals
        signals = ta.scan_for_signals(df)
        print(f"Latest signals: {signals}")
        
        # Plot chart
        ta.plot_chart(df, 
                     indicators=['sma_20', 'sma_50', 'bb_upper', 'bb_lower'], 
                     title='Cardano 1h Chart with Indicators',
                     save_path='data/cardano/charts/latest_analysis.png')
