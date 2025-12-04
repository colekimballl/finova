#!/usr/bin/env python
'''
Cardano Data Visualizer
-----------------------
Visualizes Cardano price data across different timeframes.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
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
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        print(f"Loaded {len(df)} rows of {timeframe} data")
        return df
        
    except Exception as e:
        print(f"Error loading {timeframe} data: {str(e)}")
        return None

def plot_price_history(df, timeframe, days=None, save_dir=None):
    """
    Plot the price history for a timeframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    timeframe : str
        Timeframe for the data
    days : int
        Number of days to plot (None for all data)
    save_dir : str
        Directory to save the plot (None to display only)
    """
    if df is None or df.empty:
        print(f"No data available for {timeframe} timeframe")
        return
    
    # Filter the data if days is specified
    if days is not None:
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df.index >= start_date]
    
    # Create figure and axes
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price chart
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.set_title(f'Cardano (ADA-USD) Price History - {timeframe.upper()} Timeframe')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis dates
    if len(df) > 0:
        if timeframe in ['1m', '5m', '15m']:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Volume chart
    ax2 = axes[1]
    ax2.bar(df.index, df['volume'], color='green', alpha=0.5, label='Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Price change percentage chart
    ax3 = axes[2]
    df['pct_change'] = df['close'].pct_change() * 100
    ax3.plot(df.index, df['pct_change'], color='red', label='Daily % Change')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax3.set_ylabel('Price Change (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        days_suffix = f"_{days}days" if days is not None else ""
        save_path = os.path.join(save_dir, f"price_history_{timeframe}{days_suffix}.png")
        plt.savefig(save_path)
        print(f"Price history plot saved to {save_path}")
    else:
        plt.show()

def plot_combined_timeframes(days=30, save_dir=None):
    """
    Create a combined plot showing multiple timeframes.
    
    Parameters:
    -----------
    days : int
        Number of days to plot
    save_dir : str
        Directory to save the plot (None to display only)
    """
    # Define the timeframes to plot
    timeframes = ['1h', '4h', '1d']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(timeframes), 1, figsize=(12, 12), sharex=False)
    
    # Plot each timeframe
    for i, timeframe in enumerate(timeframes):
        df = load_data(timeframe)
        if df is None or df.empty:
            continue
        
        # Filter data for the specified number of days
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=days)
        df_filtered = df[df.index >= start_date]
        
        if df_filtered.empty:
            continue
        
        # Plot the data
        axes[i].plot(df_filtered.index, df_filtered['close'], label=f'{timeframe} Close')
        axes[i].set_title(f'ADA-USD - {timeframe.upper()} Timeframe')
        axes[i].set_ylabel('Price (USD)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Format x-axis dates
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"combined_timeframes_{days}days.png")
        plt.savefig(save_path)
        print(f"Combined timeframes plot saved to {save_path}")
    else:
        plt.show()

def create_candlestick_chart(df, timeframe, days=30, save_dir=None):
    """
    Create a candlestick chart for a timeframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price data
    timeframe : str
        Timeframe for the data
    days : int
        Number of days to plot
    save_dir : str
        Directory to save the plot (None to display only)
    """
    if df is None or df.empty:
        print(f"No data available for {timeframe} timeframe")
        return
    
    # Filter the data if days is specified
    if days is not None:
        end_date = df.index.max()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df.index >= start_date]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine the appropriate candlestick width
    if timeframe == '1d':
        width = 0.6
    elif timeframe == '4h':
        width = 0.4
    elif timeframe == '1h':
        width = 0.2
    else:
        width = 0.1
    
    # Create candlestick chart
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['close'] >= row['open']:
            color = 'green'
        else:
            color = 'red'
        
        # Plot the candle body
        ax.bar(i, row['close'] - row['open'], bottom=row['open'], color=color, width=width, alpha=0.5)
        
        # Plot the wicks
        ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
    
    # Set x-ticks to datetime values
    step = max(1, len(df) // 20)  # Show roughly 20 tick labels
    x_ticks = range(0, len(df), step)
    x_labels = [df.index[i].strftime('%Y-%m-%d') if timeframe == '1d' else df.index[i].strftime('%m-%d %H:%M') for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    
    # Set labels and title
    ax.set_title(f'Cardano (ADA-USD) Candlestick Chart - {timeframe.upper()} Timeframe')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, f"candlestick_{timeframe}_{days}days.png")
        plt.savefig(save_path)
        print(f"Candlestick chart saved to {save_path}")
    else:
        plt.show()

def analyze_data_summary(save_dir=None):
    """
    Create a summary of all available data and visualize it.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save the summary report and plots
    """
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    timeframe_data = {}
    
    # Collect data for each timeframe
    for tf in timeframes:
        df = load_data(tf)
        if df is not None:
            timeframe_data[tf] = {
                'rows': len(df),
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'duration_days': (df.index.max() - df.index.min()).days,
                'min_price': df['low'].min(),
                'max_price': df['high'].max(),
                'last_price': df['close'].iloc[-1],
                'price_change': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                'avg_volume': df['volume'].mean()
            }
    
    # Print summary
    print("\n" + "="*80)
    print(" DATA SUMMARY ")
    print("="*80)
    
    for tf, data in timeframe_data.items():
        print(f"\nTimeframe: {tf}")
        print(f"  Rows: {data['rows']}")
        print(f"  Date Range: {data['start_date'].date()} to {data['end_date'].date()} ({data['duration_days']} days)")
        print(f"  Price Range: ${data['min_price']:.4f} to ${data['max_price']:.4f}")
        print(f"  Last Price: ${data['last_price']:.4f}")
        print(f"  Overall Price Change: {data['price_change']:.2f}%")
        print(f"  Average Volume: {data['avg_volume']:.2f}")
    
    # Create a bar chart of total rows by timeframe
    fig, ax = plt.subplots(figsize=(10, 6))
    timeframe_labels = list(timeframe_data.keys())
    row_counts = [data['rows'] for data in timeframe_data.values()]
    
    ax.bar(timeframe_labels, row_counts, color='blue', alpha=0.7)
    ax.set_title('Number of Data Points by Timeframe')
    ax.set_xlabel('Timeframe')
    ax.set_ylabel('Number of Data Points')
    
    # Add data labels on top of the bars
    for i, count in enumerate(row_counts):
        ax.text(i, count + (max(row_counts) * 0.01), f"{count}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, "data_summary_barplot.png")
        plt.savefig(save_path)
        print(f"Data summary bar plot saved to {save_path}")
        
        # Also save a text summary
        summary_path = os.path.join(save_dir, "data_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("CARDANO PRICE DATA SUMMARY\n")
            f.write("==========================\n\n")
            
            for tf, data in timeframe_data.items():
                f.write(f"Timeframe: {tf}\n")
                f.write(f"  Rows: {data['rows']}\n")
                f.write(f"  Date Range: {data['start_date'].date()} to {data['end_date'].date()} ({data['duration_days']} days)\n")
                f.write(f"  Price Range: ${data['min_price']:.4f} to ${data['max_price']:.4f}\n")
                f.write(f"  Last Price: ${data['last_price']:.4f}\n")
                f.write(f"  Overall Price Change: {data['price_change']:.2f}%\n")
                f.write(f"  Average Volume: {data['avg_volume']:.2f}\n\n")
        
        print(f"Data summary text saved to {summary_path}")
    else:
        plt.show()

def generate_all_visualizations(output_dir='../data/processed'):
    """
    Generate all visualizations for the dataset.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save all outputs
    """
    # Create output directories
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create data summary
    analyze_data_summary(save_dir=viz_dir)
    
    # Create price history plots for each timeframe
    timeframes = ['1h', '4h', '1d']
    for tf in timeframes:
        df = load_data(tf)
        if df is not None:
            # Plot the full history
            plot_price_history(df, tf, save_dir=viz_dir)
            
            # Plot the recent history (last 30 days)
            plot_price_history(df, tf, days=30, save_dir=viz_dir)
            
            # Create candlestick charts
            create_candlestick_chart(df, tf, days=30, save_dir=viz_dir)
    
    # Create combined timeframe plot
    plot_combined_timeframes(days=30, save_dir=viz_dir)
    
    print(f"\nAll visualizations saved to {viz_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cardano Data Visualizer')
    parser.add_argument('--timeframe', type=str, default='1h', 
                      choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                      help='Timeframe to visualize (default: 1h)')
    parser.add_argument('--days', type=int, default=None,
                      help='Number of days to plot (default: all available)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for saved visualizations')
    parser.add_argument('--all', action='store_true',
                      help='Generate all visualizations')
    parser.add_argument('--candlestick', action='store_true',
                      help='Create a candlestick chart')
    parser.add_argument('--summary', action='store_true',
                      help='Show data summary')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_visualizations(args.output)
    elif args.summary:
        analyze_data_summary(args.output)
    elif args.candlestick:
        df = load_data(args.timeframe)
        create_candlestick_chart(df, args.timeframe, days=args.days, save_dir=args.output)
    else:
        df = load_data(args.timeframe)
        plot_price_history(df, args.timeframe, days=args.days, save_dir=args.output)
