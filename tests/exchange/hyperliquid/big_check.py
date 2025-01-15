import os
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
from dotenv import load_dotenv
from termcolor import cprint
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketAnalyzer:
    def __init__(self):
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        # List of symbols to analyze; add or remove symbols as needed
        self.symbols = ["BTC", "ETH", "ADA", "SOL", "DOGE", "XRP", "LTC", "BNB"]
        self.data_dir = Path("market_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Alert thresholds (in percentages)
        self.price_alerts = {
            'BTC': {'upper': 2.0, 'lower': -2.0},
            'ETH': {'upper': 3.0, 'lower': -3.0},
            'SOL': {'upper': 5.0, 'lower': -5.0},
            'ADA': {'upper': 4.0, 'lower': -4.0},
            'DOGE': {'upper': 6.0, 'lower': -6.0},
            'XRP': {'upper': 3.5, 'lower': -3.5},
            'LTC': {'upper': 4.0, 'lower': -4.0},
            'BNB': {'upper': 3.0, 'lower': -3.0}
        }

    def fetch_historical_data(self, symbol, lookback_hours=72):
        """Fetch and process historical trade data for a given symbol."""
        url = "https://api.hyperliquid.xyz/info"
        headers = {"Content-Type": "application/json"}
        data = {"type": "trades", "coin": symbol}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code != 200:
                logging.error(f"Failed to fetch trades for {symbol}. Status Code: {response.status_code}")
                return pd.DataFrame()
            
            trades = response.json()
            if not isinstance(trades, list) or not trades:
                logging.warning(f"No trade data returned for {symbol}.")
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            # Ensure required columns are present
            required_columns = {'px', 'sz', 'time'}
            if not required_columns.issubset(df.columns):
                logging.error(f"Missing required columns in trade data for {symbol}. Received columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            df['px'] = pd.to_numeric(df['px'], errors='coerce')
            df['sz'] = pd.to_numeric(df['sz'], errors='coerce')
            df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            df.dropna(subset=['px', 'sz', 'time'], inplace=True)
            df = df.sort_values('time').reset_index(drop=True)
            
            # Filter data within the lookback period
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            df = df[df['time'] >= cutoff_time]
            logging.info(f"Fetched {len(df)} trades for {symbol} within the last {lookback_hours} hours.")
            return df
        except Exception as e:
            logging.exception(f"Exception occurred while fetching trades for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_volatility_indicators(self, df):
        """Calculate basic volatility metrics."""
        if len(df) < 2:
            return {'realized_vol': np.nan, 'range_vol': np.nan, 'recent_vol': np.nan}
        
        returns = df['px'].pct_change().dropna()
        high_low_range = df['px'].max() - df['px'].min()
        
        return {
            'realized_vol': returns.std() * np.sqrt(365 * 24),  # Annualized volatility
            'range_vol': high_low_range / df['px'].mean(),
            'recent_vol': returns.tail(50).std() * np.sqrt(365 * 24)
        }

    def analyze_volume_profile(self, df):
        """Analyze volume distribution and patterns."""
        if len(df) < 2:
            return {'total_volume': np.nan, 'avg_trade_size': np.nan, 'volume_concentration': np.nan, 'price_level_max_vol': np.nan}
        
        try:
            price_buckets = pd.qcut(df['px'], q=10, duplicates='drop')
        except ValueError:
            # If there are not enough unique prices, use equal-width bins
            price_buckets = pd.cut(df['px'], bins=10)
        
        volume_profile = df.groupby(price_buckets)['sz'].sum()
        
        return {
            'total_volume': df['sz'].sum(),
            'avg_trade_size': df['sz'].mean(),
            'volume_concentration': volume_profile.max() / volume_profile.sum() if volume_profile.sum() != 0 else np.nan,
            'price_level_max_vol': volume_profile.idxmax().left if hasattr(volume_profile.idxmax(), 'left') else volume_profile.idxmax()
        }

    def predict_funding_rate(self, symbol, historical_rates):
        """Simple funding rate prediction based on recent trends."""
        if len(historical_rates) < 2:
            return {'next_predicted': np.nan, 'trend': 'Unknown', 'volatility': np.nan, 'confidence': 'Unknown'}
        
        rates = pd.Series(historical_rates)
        trend = rates.diff().mean()
        volatility = rates.std()
        
        prediction = {
            'next_predicted': rates.iloc[-1] + trend,
            'trend': 'Increasing' if trend > 0 else 'Decreasing',
            'volatility': volatility,
            'confidence': 'High' if volatility < abs(trend) else 'Low'
        }
        return prediction

    def calculate_order_book_metrics(self, l2_data):
        """Calculate spread and order book imbalance."""
        if len(l2_data) < 2:
            return {'spread': np.nan, 'order_book_imbalance': np.nan}
        
        bids = l2_data[0]
        asks = l2_data[1]
        
        if not bids or not asks:
            return {'spread': np.nan, 'order_book_imbalance': np.nan}
        
        try:
            best_bid = float(bids[0]["px"])
            best_ask = float(asks[0]["px"])
            spread = best_ask - best_bid
            
            total_bid_volume = sum(float(bid["sz"]) for bid in bids)
            total_ask_volume = sum(float(ask["sz"]) for ask in asks)
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) != 0 else np.nan
            
            return {
                'spread': spread,
                'order_book_imbalance': imbalance
            }
        except Exception as e:
            logging.exception(f"Error calculating order book metrics: {e}")
            return {'spread': np.nan, 'order_book_imbalance': np.nan}

    def check_price_alerts(self, symbol, current_price, reference_price):
        """Check if price movements trigger any alerts."""
        if symbol not in self.price_alerts:
            return []
        
        if reference_price == 0:
            logging.warning(f"Reference price for {symbol} is zero. Cannot calculate percentage change.")
            return []
        
        pct_change = ((current_price - reference_price) / reference_price) * 100
        alerts = []
        
        if pct_change >= self.price_alerts[symbol]['upper']:
            alerts.append(f"🔼 {symbol} up {pct_change:.2f}% - Upper threshold hit")
        elif pct_change <= self.price_alerts[symbol]['lower']:
            alerts.append(f"🔽 {symbol} down {pct_change:.2f}% - Lower threshold hit")
        
        return alerts

    def analyze_markets(self):
        try:
            market_data = []
            all_alerts = []
            
            for symbol in self.symbols:
                try:
                    # Fetch historical trade data
                    historical_df = self.fetch_historical_data(symbol)
                    
                    if historical_df.empty:
                        logging.warning(f"No historical data for {symbol}. Skipping.")
                        continue
                    
                    # Get current price and order book
                    url = "https://api.hyperliquid.xyz/info"
                    headers = {"Content-Type": "application/json"}
                    l2_response = requests.post(url, headers=headers, json={"type": "l2Book", "coin": symbol})
                    
                    if l2_response.status_code != 200:
                        logging.error(f"Failed to fetch order book for {symbol}. Status Code: {l2_response.status_code}")
                        current_price = np.nan
                        l2_data = []
                    else:
                        l2_data = l2_response.json().get("levels", [])
                        if len(l2_data) >= 2 and l2_data[0] and l2_data[1]:
                            try:
                                best_bid = float(l2_data[0][0]["px"])
                                best_ask = float(l2_data[1][0]["px"])
                                current_price = (best_bid + best_ask) / 2
                            except (KeyError, IndexError, ValueError) as e:
                                logging.exception(f"Error parsing order book data for {symbol}: {e}")
                                current_price = np.nan
                        else:
                            logging.warning(f"Insufficient order book data for {symbol}.")
                            current_price = np.nan
                    
                    # Calculate metrics
                    vol_indicators = self.calculate_volatility_indicators(historical_df)
                    volume_profile = self.analyze_volume_profile(historical_df)
                    order_book_metrics = self.calculate_order_book_metrics(l2_data)
                    funding_pred = self.predict_funding_rate(symbol, [0.01, 0.02, 0.015])  # Example rates
                    
                    # Check alerts
                    if not historical_df.empty and not np.isnan(current_price):
                        reference_price = historical_df['px'].iloc[0]
                        alerts = self.check_price_alerts(symbol, current_price, reference_price)
                        if alerts:
                            all_alerts.extend(alerts)
                    
                    market_data.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'volatility': vol_indicators.get('realized_vol', np.nan),
                        'recent_volatility': vol_indicators.get('recent_vol', np.nan),
                        'range_volatility': vol_indicators.get('range_vol', np.nan),
                        'total_volume': volume_profile.get('total_volume', np.nan),
                        'avg_trade_size': volume_profile.get('avg_trade_size', np.nan),
                        'volume_concentration': volume_profile.get('volume_concentration', np.nan),
                        'price_level_max_vol': volume_profile.get('price_level_max_vol', np.nan),
                        'spread': order_book_metrics.get('spread', np.nan),
                        'order_book_imbalance': order_book_metrics.get('order_book_imbalance', np.nan),
                        'volume_over_1h': self.calculate_volume_over_period(historical_df, hours=1),
                        'volume_over_6h': self.calculate_volume_over_period(historical_df, hours=6),
                        'volume_over_12h': self.calculate_volume_over_period(historical_df, hours=12),
                        'predicted_funding': funding_pred.get('next_predicted', np.nan),
                        'funding_confidence': funding_pred.get('confidence', 'Unknown'),
                    })
                    
                except Exception as e:
                    cprint(f"\n❌ Error analyzing {symbol}: {e}", "red")
                    logging.exception(f"Error analyzing {symbol}")
                    continue
            
            if not market_data:
                cprint("\n❌ No market data available to display.", "red")
                return
            
            # Create DataFrame and display results
            df = pd.DataFrame(market_data)
            
            cprint("\n📊 Advanced Market Analysis", "cyan", attrs=["bold"])
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\nPrice & Volatility:")
            try:
                print(df[['symbol', 'current_price', 'volatility', 'recent_volatility', 'range_volatility']].to_string(index=False))
            except KeyError as e:
                logging.error(f"Missing columns for Price & Volatility section: {e}")
            
            print("\nVolume & Funding Analysis:")
            try:
                print(df[['symbol', 'total_volume', 'avg_trade_size', 'volume_concentration', 'price_level_max_vol',
                          'volume_over_1h', 'volume_over_6h', 'volume_over_12h',
                          'predicted_funding', 'funding_confidence']].to_string(index=False))
            except KeyError as e:
                logging.error(f"Missing columns for Volume & Funding Analysis section: {e}")
            
            print("\nOrder Book Metrics:")
            try:
                print(df[['symbol', 'spread', 'order_book_imbalance']].to_string(index=False))
            except KeyError as e:
                logging.error(f"Missing columns for Order Book Metrics section: {e}")
            
            if all_alerts:
                cprint("\n⚠️ Price Alerts:", "yellow", attrs=["bold"])
                for alert in all_alerts:
                    print(alert)
            
            # Save to CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = self.data_dir / f'advanced_market_data_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            cprint(f"\n💾 Data saved to: {csv_path}", "green")
            
            return df
            
        except Exception as e:
            cprint(f"\n❌ Error: {str(e)}", "red")
            logging.exception("An error occurred during market analysis.")
            return None

    def calculate_volume_over_period(self, df, hours=1):
        """Calculate total volume over the last specified hours."""
        if df.empty:
            return np.nan
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_df = df[df['time'] >= cutoff_time]
        return recent_df['sz'].sum()

if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    df = analyzer.analyze_markets()

