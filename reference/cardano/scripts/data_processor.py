#!/usr/bin/env python
'''
Cardano Data Processor
---------------------
Processes raw Cardano price data into enhanced datasets with various indicators
and features useful for analysis and machine learning.
'''

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import argparse
from pathlib import Path

def load_data(timeframe, data_dir='../data/cardano'):
    """
    Load data for a specific timeframe.
    
    Parameters:
    -----------
    timeframe : str
        Timeframe to load ('1m', '5m', '15m', '1h', '4h', '1d')
    data_dir : str
        Directory containing the data files
    
    Returns:
    --------
    pd.DataFrame : Loaded data or None if file not found
    """
    try:
        # Construct the filepath
        if timeframe == '1m':
            filepath = os.path.join(data_dir, f'ADAUSD-{timeframe}-data.csv')
        else:
            filepath = os.path.join(data_dir, f'ADAUSD-{timeframe}.csv')
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Error: Data file for {timeframe} timeframe not found at {filepath}")
            return None
        
        # Load the data
        df = pd.read_csv(filepath)
        
        # Convert datetime column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"Loaded {len(df)} rows of {timeframe} data")
        return df
        
    except Exception as e:
        print(f"Error loading {timeframe} data: {str(e)}")
        return None

def add_basic_features(df):
    """
    Add basic features to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set datetime as index for easier calculation
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Basic price features
    df['price_range'] = df['high'] - df['low']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Calculate returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility features
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['relative_volume'] = df['volume'] / df['volume_ma']
    
    # Reset index to get datetime as a column again
    df.reset_index(inplace=True)
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set datetime as index for calculations
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        if len(df) >= window:
            df[f'sma_{window}'] = ta.sma(df['close'], length=window)
            df[f'ema_{window}'] = ta.ema(df['close'], length=window)
    
    # MACD
    if len(df) >= 26:
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
    
    # RSI
    if len(df) >= 14:
        df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # Bollinger Bands
    if len(df) >= 20:
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic Oscillator
    if len(df) >= 14:
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Average True Range (ATR)
    if len(df) >= 14:
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Commodity Channel Index (CCI)
    if len(df) >= 20:
        df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # Reset index to get datetime as a column again
    df.reset_index(inplace=True)
    
    return df

def add_pattern_recognition(df):
    """
    Add candlestick pattern recognition features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added pattern features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set datetime as index for calculations
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Define functions to identify common candlestick patterns
    
    # Doji
    df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
    
    # Hammer
    df['hammer'] = (
        ((df['high'] - df[['open', 'close']].max(axis=1)) < 
         (0.3 * (df['high'] - df['low']))) &
        ((df[['open', 'close']].min(axis=1) - df['low']) > 
         (2 * (df[['open', 'close']].max(axis=1) - df[['open', 'close']].min(axis=1))))
    ).astype(int)
    
    # Shooting Star
    df['shooting_star'] = (
        ((df[['open', 'close']].min(axis=1) - df['low']) < 
         (0.3 * (df['high'] - df['low']))) &
        ((df['high'] - df[['open', 'close']].max(axis=1)) > 
         (2 * (df[['open', 'close']].max(axis=1) - df[['open', 'close']].min(axis=1))))
    ).astype(int)
    
    # Bullish Engulfing
    df['bullish_engulfing'] = (
        (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is red
        (df['close'] > df['open']) &  # Current candle is green
        (df['open'] <= df['close'].shift(1)) &  # Open below previous close
        (df['close'] >= df['open'].shift(1))  # Close above previous open
    ).astype(int)
    
    # Bearish Engulfing
    df['bearish_engulfing'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is green
        (df['open'] > df['close']) &  # Current candle is red
        (df['open'] >= df['close'].shift(1)) &  # Open above previous close
        (df['close'] <= df['open'].shift(1))  # Close below previous open
    ).astype(int)
    
    # Reset index to get datetime as a column again
    df.reset_index(inplace=True)
    
    return df

def add_market_regime_features(df):
    """
    Add market regime identification features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added market regime features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set datetime as index for calculations
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Calculate 50-day and 200-day moving averages if enough data
    if len(df) >= 200:
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # Define market trend based on moving average positions
        df['ma_trend'] = 'neutral'
        df.loc[df['sma_50'] > df['sma_200'], 'ma_trend'] = 'bullish'
        df.loc[df['sma_50'] < df['sma_200'], 'ma_trend'] = 'bearish'
        
        # Convert to numeric for machine learning
        df['ma_trend_numeric'] = 0  # neutral
        df.loc[df['ma_trend'] == 'bullish', 'ma_trend_numeric'] = 1
        df.loc[df['ma_trend'] == 'bearish', 'ma_trend_numeric'] = -1
    
    # Calculate volatility regimes
    if len(df) >= 20:
        # 20-day rolling standard deviation of returns
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        
        # Define volatility regimes based on quantiles
        vol_quantiles = df['volatility_20d'].quantile([0.33, 0.66]).values
        
        df['volatility_regime'] = 'medium'
        df.loc[df['volatility_20d'] <= vol_quantiles[0], 'volatility_regime'] = 'low'
        df.loc[df['volatility_20d'] > vol_quantiles[1], 'volatility_regime'] = 'high'
        
        # Convert to numeric for machine learning
        df['volatility_regime_numeric'] = 0  # medium
        df.loc[df['volatility_regime'] == 'low', 'volatility_regime_numeric'] = -1
        df.loc[df['volatility_regime'] == 'high', 'volatility_regime_numeric'] = 1
    
    # Reset index to get datetime as a column again
    df.reset_index(inplace=True)
    
    return df

def generate_ml_features(df):
    """
    Generate features specifically for machine learning models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added ML features
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set datetime as index for calculations
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Generate target variables for different prediction horizons
    for horizon in [1, 3, 5, 10]:
        if len(df) > horizon:
            # Price change over the next N periods
            df[f'target_return_{horizon}'] = df['close'].pct_change(periods=horizon).shift(-horizon)
            
            # Binary target (1 if price went up, 0 if down)
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
    
    # Generate lagged features
    for lag in [1, 2, 3, 5]:
        if len(df) > lag:
            # Price lags
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return'].shift(lag)
            
            # Volume lags
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # Price change from N periods ago
            df[f'price_change_{lag}'] = (df['close'] / df[f'close_lag_{lag}']) - 1
    
    # Create rolling window features
    for window in [5, 10, 20]:
        if len(df) >= window:
            # Rolling mean of returns
            df[f'return_mean_{window}'] = df['return'].rolling(window=window).mean()
            
            # Rolling standard deviation of returns (volatility)
            df[f'return_std_{window}'] = df['return'].rolling(window=window).std()
            
            # Rolling min/max
            df[f'price_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'price_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Percentage distance from rolling min/max
            df[f'pct_from_min_{window}'] = (df['close'] / df[f'price_min_{window}']) - 1
            df[f'pct_from_max_{window}'] = (df['close'] / df[f'price_max_{window}']) - 1
    
    # Reset index to get datetime as a column again
    df.reset_index(inplace=True)
    
    return df

def process_data(df, add_features=True, add_indicators=True, add_patterns=True, 
                add_regimes=True, add_ml=True):
    """
    Process the data by adding various feature sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    add_features : bool
        Whether to add basic features
    add_indicators : bool
        Whether to add technical indicators
    add_patterns : bool
        Whether to add candlestick patterns
    add_regimes : bool
        Whether to add market regime features
    add_ml : bool
        Whether to add machine learning features
    
    Returns:
    --------
    pd.DataFrame : Processed DataFrame
    """
    if df is None or df.empty:
        print("No data to process")
        return None
    
    # Make a copy of the original data
    processed_df = df.copy()
    
    # Apply each processing step if requested
    if add_features:
        print("Adding basic features...")
        processed_df = add_basic_features(processed_df)
    
    if add_indicators:
        print("Adding technical indicators...")
        processed_df = add_technical_indicators(processed_df)
    
    if add_patterns:
        print("Adding candlestick patterns...")
        processed_df = add_pattern_recognition(processed_df)
    
    if add_regimes:
        print("Adding market regime features...")
        processed_df = add_market_regime_features(processed_df)
    
    if add_ml:
        print("Adding machine learning features...")
        processed_df = generate_ml_features(processed_df)
    
    # Fill NaN values - this is necessary because many indicators create NaN values at the beginning
    processed_df = processed_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return processed_df

def save_processed_data(df, timeframe, output_dir='../data/processed'):
    """
    Save processed data to output directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed DataFrame to save
    timeframe : str
        Timeframe identifier for the filename
    output_dir : str
        Directory to save processed data
    """
    if df is None or df.empty:
        print("No data to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    filename = f"ADAUSD-{timeframe}-processed.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save the data as CSV
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")
    
    # Try to save as parquet if the libraries are available
    try:
        parquet_filepath = filepath.replace('.csv', '.parquet')
        df.to_parquet(parquet_filepath, index=False)
        print(f"Processed data also saved as parquet to {parquet_filepath}")
    except ImportError:
        print("Parquet libraries (pyarrow or fastparquet) not installed. Skipping parquet output.")
        print("To enable parquet output, install pyarrow with: pip install pyarrow")
    except Exception as e:
        print(f"Error saving to parquet format: {str(e)}")

def process_all_timeframes(output_dir='../data/processed'):
    """
    Process data for all available timeframes.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save processed data
    """
    timeframes = ['1h', '4h', '1d']  # Focus on the most useful timeframes
    
    for timeframe in timeframes:
        print(f"\nProcessing {timeframe} timeframe data...")
        df = load_data(timeframe)
        if df is not None:
            processed_df = process_data(df, add_features=True, add_indicators=True, 
                                      add_patterns=True, add_regimes=True, add_ml=True)
            save_processed_data(processed_df, timeframe, output_dir)

def create_dataset_statistics(output_dir='../data/processed'):
    """
    Create statistical summary of processed datasets.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing processed data
    """
    # Find all processed CSV files
    processed_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('-processed.csv'):
                processed_files.append(os.path.join(root, file))
    
    if not processed_files:
        print("No processed data files found")
        return
    
    # Create a report
    report_path = os.path.join(output_dir, "dataset_statistics.txt")
    
    with open(report_path, 'w') as f:
        f.write("PROCESSED DATASET STATISTICS\n")
        f.write("===========================\n\n")
        
        for file_path in processed_files:
            try:
                # Load the dataset
                df = pd.read_csv(file_path)
                
                # Extract timeframe from filename
                filename = os.path.basename(file_path)
                timeframe = filename.split('-')[1]
                
                # Write dataset information
                f.write(f"Timeframe: {timeframe}\n")
                f.write(f"File: {filename}\n")
                f.write(f"Rows: {len(df)}\n")
                f.write(f"Columns: {len(df.columns)}\n")
                
                # Date range
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    f.write(f"Date Range: {df['datetime'].min().date()} to {df['datetime'].max().date()}\n")
                
                # List of columns by category
                basic_columns = [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume', 'datetime', 'timestamp']]
                feature_columns = [col for col in df.columns if any(x in col for x in ['price_', 'body_', 'wick', 'return', 'volatility', 'volume_'])]
                indicator_columns = [col for col in df.columns if any(x in col for x in ['sma_', 'ema_', 'macd', 'rsi', 'bb_', 'stoch', 'atr', 'cci'])]
                pattern_columns = [col for col in df.columns if any(x in col for x in ['doji', 'hammer', 'engulfing', 'star'])]
                regime_columns = [col for col in df.columns if any(x in col for x in ['trend', 'regime'])]
                ml_columns = [col for col in df.columns if any(x in col for x in ['target_', '_lag_', 'from_min', 'from_max'])]
                
                f.write("\nColumn Categories:\n")
                f.write(f"  Basic Columns: {len(basic_columns)}\n")
                f.write(f"  Feature Columns: {len(feature_columns)}\n")
                f.write(f"  Technical Indicator Columns: {len(indicator_columns)}\n")
                f.write(f"  Pattern Recognition Columns: {len(pattern_columns)}\n")
                f.write(f"  Market Regime Columns: {len(regime_columns)}\n")
                f.write(f"  Machine Learning Columns: {len(ml_columns)}\n")
                
                # Basic statistics for key columns
                if 'close' in df.columns:
                    f.write(f"\nPrice Statistics:\n")
                    f.write(f"  Min: ${df['close'].min():.4f}\n")
                    f.write(f"  Max: ${df['close'].max():.4f}\n")
                    f.write(f"  Mean: ${df['close'].mean():.4f}\n")
                    f.write(f"  Current: ${df['close'].iloc[-1]:.4f}\n")
                
                if 'return' in df.columns:
                    f.write(f"\nReturn Statistics:\n")
                    f.write(f"  Mean Return: {df['return'].mean():.6f}\n")
                    f.write(f"  Return Std Dev: {df['return'].std():.6f}\n")
                    f.write(f"  Max Return: {df['return'].max():.4f}\n")
                    f.write(f"  Min Return: {df['return'].min():.4f}\n")
                
                f.write("\n" + "="*50 + "\n\n")
                
            except Exception as e:
                f.write(f"Error processing {filename}: {str(e)}\n\n")
    
    print(f"Dataset statistics report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cardano Data Processor')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                      help='Timeframe to process (default: 1h)')
    parser.add_argument('--output', type=str, default='../data/processed',
                      help='Output directory for processed data')
    parser.add_argument('--features', action='store_true',
                      help='Add basic features')
    parser.add_argument('--indicators', action='store_true',
                      help='Add technical indicators')
    parser.add_argument('--patterns', action='store_true',
                      help='Add candlestick patterns')
    parser.add_argument('--regimes', action='store_true',
                      help='Add market regime features')
    parser.add_argument('--ml', action='store_true',
                      help='Add machine learning features')
    parser.add_argument('--all-features', action='store_true',
                      help='Add all feature types')
    parser.add_argument('--all-timeframes', action='store_true',
                      help='Process all timeframes')
    parser.add_argument('--stats', action='store_true',
                      help='Generate dataset statistics')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.all_timeframes:
        process_all_timeframes(args.output)
    else:
        # Determine which features to add
        add_features = args.features or args.all_features
        add_indicators = args.indicators or args.all_features
        add_patterns = args.patterns or args.all_features
        add_regimes = args.regimes or args.all_features
        add_ml = args.ml or args.all_features
        
        # If no feature flags are specified, add basic features by default
        if not any([add_features, add_indicators, add_patterns, add_regimes, add_ml]):
            add_features = True
        
        # Load and process data
        df = load_data(args.timeframe)
        if df is not None:
            processed_df = process_data(df, add_features, add_indicators, add_patterns, add_regimes, add_ml)
            save_processed_data(processed_df, args.timeframe, args.output)
    
    if args.stats:
        create_dataset_statistics(args.output)
    
    print("\nData processing complete!")
