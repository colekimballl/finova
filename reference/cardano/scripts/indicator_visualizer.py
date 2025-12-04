#!/usr/bin/env python
'''
Cardano Indicator Visualizer
---------------------------
Visualizes processed Cardano price data with technical indicators.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from pathlib import Path
import numpy as np

def load_processed_data(timeframe, data_dir='../data/processed'):
    """
    Load processed data for a specific timeframe.
    
    Parameters:
    -----------
    timeframe : str
        Timeframe to load ('1m', '5m', '15m', '1h', '4h', '1d')
    data_dir : str
        Directory containing the processed data files
    
    Returns:
    --------
    pd.DataFrame : Loaded data or None if file not found
    """
    try:
        filepath = os.path.join(data_dir, f'ADAUSD-{timeframe}-processed.csv')
        if not os.path.exists(filepath):
            print(f"Error: Processed data file for {timeframe} timeframe not found at {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        print(f"Loaded {len(df)} rows of processed {timeframe} data with {len(df.columns)} features")
        return df
    except Exception as e:
        print(f"Error loading processed {timeframe} data: {str(e)}")
        return None

def plot_candlestick_with_indicators(df, timeframe, days=30, save_dir=None):
    """
    Create a candlestick chart with indicators and volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing processed price data
    timeframe : str
        Timeframe for the data
    days : int
        Number of days to plot
    save_dir : str
        Directory to save the plot (None to display only)
    """
    if df is None or df.empty:
        print(f"No processed data available for {timeframe} timeframe")
        return
    
    # Check if volume data is available
    has_volume = 'volume' in df.columns
    
    # Filter the data if days is specified
    if days is not None:
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df.index >= start_date]
    
    # Reset index to iterate through rows
    df_reset = df.reset_index()
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Determine the appropriate candlestick width
    width = 0.2 if timeframe == '1h' else 0.4
    
    # If volume data is available, plot it as background
    if has_volume:
        # Normalize volume to make the plot more readable
        volume = df_reset['volume'].values
        
        # Create 5 different shades of green based on volume percentiles
        # Normalize volume between 0 and 1 for easier categorization
        if len(volume) > 0:
            # Calculate volume percentiles for 5 categories
            percentiles = np.percentile(volume, [20, 40, 60, 80])
            
            # Create colors based on volume percentiles (all green with different intensities)
            volume_colors = []
            for v in volume:
                if v <= percentiles[0]:
                    # Very low volume - very light green
                    volume_colors.append((0.8, 1, 0.8, 0.25))  # Lightest green
                elif v <= percentiles[1]:
                    # Low volume - light green
                    volume_colors.append((0.6, 0.9, 0.6, 0.25))  # Light green
                elif v <= percentiles[2]:
                    # Medium volume - medium green
                    volume_colors.append((0.4, 0.8, 0.4, 0.25))  # Medium green
                elif v <= percentiles[3]:
                    # High volume - dark green
                    volume_colors.append((0.2, 0.6, 0.2, 0.25))  # Dark green
                else:
                    # Very high volume - very dark green
                    volume_colors.append((0.0, 0.4, 0.0, 0.25))  # Darkest green
        else:
            volume_colors = [(0.5, 0.8, 0.5, 0.25) for _ in range(len(df_reset))]  # Default medium green
        
        # Plot volume as background colored regions
        for i, color in enumerate(volume_colors):
            ax1.axvspan(i - width/2, i + width/2, color=color, alpha=1)
            
        # Add a legend for volume shades
        from matplotlib.patches import Patch
        volume_legend_elements = [
            Patch(facecolor=(0.8, 1, 0.8, 0.25), label='Very Low Volume'),
            Patch(facecolor=(0.6, 0.9, 0.6, 0.25), label='Low Volume'),
            Patch(facecolor=(0.4, 0.8, 0.4, 0.25), label='Medium Volume'),
            Patch(facecolor=(0.2, 0.6, 0.2, 0.25), label='High Volume'),
            Patch(facecolor=(0.0, 0.4, 0.0, 0.25), label='Very High Volume')
        ]
    
    # Create candlestick chart
    for i, (_, row) in enumerate(df_reset.iterrows()):
        # Use green for bullish (close >= open) and red for bearish (close < open) candles
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Plot the candle body
        ax1.bar(i, row['close'] - row['open'], bottom=row['open'], color=color, width=width, alpha=0.7)
        
        # Plot the wicks
        ax1.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
    
    # Add moving averages if available
    for ma in ['sma_20', 'sma_50', 'sma_200']:
        if ma in df.columns:
            ax1.plot(range(len(df_reset)), df_reset[ma], label=ma, alpha=0.7)
    
    # Add Bollinger Bands if available
    for bb in ['bb_upper', 'bb_middle', 'bb_lower']:
        if bb in df.columns:
            ax1.plot(range(len(df_reset)), df_reset[bb], label=bb, alpha=0.5)
    
    # Volume is now plotted as background colors in the main chart
    
    # Set x-ticks to datetime values
    step = max(1, len(df_reset) // 15)  # Show roughly 15 tick labels
    x_ticks = range(0, len(df_reset), step)
    x_labels = [df_reset['datetime'].iloc[i].strftime('%Y-%m-%d') 
               if timeframe == '1d' else 
               df_reset['datetime'].iloc[i].strftime('%m-%d %H:%M') 
               for i in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45)
    
    # Set labels and title
    ax1.set_title(f'Cardano (ADA-USD) Candlestick Chart - {timeframe.upper()} Timeframe')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)
    # Add both indicator legend and volume legend
    # First get handles and labels for the indicators
    handles, labels = ax1.get_legend_handles_labels()
    
    # Add volume legend elements if volume is available
    if has_volume:
        handles = handles + volume_legend_elements
        labels = labels + ['Very Low Volume', 'Low Volume', 'Medium Volume', 'High Volume', 'Very High Volume']
    
    # Add the combined legend
    ax1.legend(handles=handles, labels=labels, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"candlestick_indicators_{timeframe}_{days}days.png")
        plt.savefig(save_path)
        print(f"Candlestick chart with indicators saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Cardano Indicator Visualizer')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                      help='Timeframe to visualize (default: 1h)')
    parser.add_argument('--days', type=int, default=30,
                      help='Number of days to plot (default: 30)')
    parser.add_argument('--candlestick', action='store_true',
                      help='Create a candlestick chart with indicators')
    parser.add_argument('--output', type=str, default='../data/processed/visualizations',
                      help='Output directory for saved visualizations')
    parser.add_argument('--show', action='store_true',
                      help='Show the plot instead of saving it')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Arguments used:")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Candlestick: {args.candlestick}")
    print(f"Output Directory: {args.output}")
    print(f"Show Plot: {args.show}")
    
    # Load processed data
    df = load_processed_data(args.timeframe)
    
    if df is not None:
        if args.candlestick:
            plot_candlestick_with_indicators(df, args.timeframe, args.days, 
                                          None if args.show else args.output)
        else:
            print("No visualization type specified. Use --candlestick to plot candlesticks.")

if __name__ == "__main__":
    main()
