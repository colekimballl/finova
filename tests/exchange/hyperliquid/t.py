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
warnings.filterwarnings('ignore')

class MarketAnalyzer:
    def __init__(self):
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.symbols = ["BTC", "ETH", "ADA", "SOL", "DOGE"]
        self.data_dir = Path("market_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Alert thresholds (in percentages)
        self.price_alerts = {
            'BTC': {'upper': 2.0, 'lower': -2.0},
            'ETH': {'upper': 3.0, 'lower': -3.0},
            'SOL': {'upper': 5.0, 'lower': -5.0},
            'ADA': {'upper': 4.0, 'lower': -4.0},
            'DOGE': {'upper': 6.0, 'lower': -6.0}
        }

    def fetch_historical_data(self, symbol, lookback_hours=24):
        """Fetch and process historical data for volatility analysis"""
        url = "https://api.hyperliquid.xyz/info"
        headers = {"Content-Type": "application/json"}
        data = {"type": "trades", "coin": symbol}
        
        response = requests.post(url, headers=headers, json=data)
        trades = response.json() if response.status_code == 200 else []
        
        if trades:
            df = pd.DataFrame(trades)
            df['px'] = pd.to_numeric(df['px'])
            df['sz'] = pd.to_numeric(df['sz'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df
        return pd.DataFrame()

    def calculate_volatility_indicators(self, df):
        """Calculate various volatility metrics"""
        if len(df) < 2:
            return {}
            
        returns = df['px'].pct_change()
        high_low_range = df['px'].max() - df['px'].min()
        
        return {
            'realized_vol': returns.std() * np.sqrt(365 * 24),  # Annualized
            'range_vol': high_low_range / df['px'].mean(),
            'recent_vol': returns.tail(50).std() * np.sqrt(365 * 24)
        }

    def analyze_volume_profile(self, df):
        """Analyze volume distribution and patterns"""
        if len(df) < 2:
            return {}
            
        price_buckets = pd.qcut(df['px'], q=10)
        volume_profile = df.groupby(price_buckets)['sz'].sum()
        
        return {
            'total_volume': df['sz'].sum(),
            'avg_trade_size': df['sz'].mean(),
            'volume_concentration': volume_profile.max() / volume_profile.sum(),
            'price_level_max_vol': volume_profile.idxmax().left
        }

    def predict_funding_rate(self, symbol, historical_rates):
        """Simple funding rate prediction based on recent trends"""
        if len(historical_rates) < 2:
            return {}
            
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

    def check_price_alerts(self, symbol, current_price, reference_price):
        """Check if price movements trigger any alerts"""
        if symbol not in self.price_alerts:
            return None
            
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
                    # Fetch current market data
                    historical_df = self.fetch_historical_data(symbol)
                    
                    # Get current price and order book
                    url = "https://api.hyperliquid.xyz/info"
                    headers = {"Content-Type": "application/json"}
                    l2_response = requests.post(url, headers=headers, json={"type": "l2Book", "coin": symbol})
                    l2_data = l2_response.json().get("levels", [])
                    
                    if len(l2_data) >= 2:
                        bids = l2_data[0]
                        asks = l2_data[1]
                        current_price = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2
                        
                        # Calculate metrics
                        vol_indicators = self.calculate_volatility_indicators(historical_df)
                        volume_profile = self.analyze_volume_profile(historical_df)
                        funding_pred = self.predict_funding_rate(symbol, [0.01, 0.02, 0.015])  # Example rates
                        
                        # Check alerts
                        if len(historical_df) > 0:
                            reference_price = historical_df['px'].iloc[0]
                            alerts = self.check_price_alerts(symbol, current_price, reference_price)
                            if alerts:
                                all_alerts.extend(alerts)
                        
                        market_data.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'volatility': vol_indicators.get('realized_vol', 0),
                            'recent_volatility': vol_indicators.get('recent_vol', 0),
                            'volume_concentration': volume_profile.get('volume_concentration', 0),
                            'predicted_funding': funding_pred.get('next_predicted', 0),
                            'funding_confidence': funding_pred.get('confidence', 'Unknown'),
                        })
                        
                except Exception as e:
                    cprint(f"\n❌ Error analyzing {symbol}: {e}", "red")
                    continue
            
            # Create DataFrame and display results
            df = pd.DataFrame(market_data)
            
            cprint("\n📊 Advanced Market Analysis", "cyan", attrs=["bold"])
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("\nPrice & Volatility:")
            print(df[['symbol', 'current_price', 'volatility', 'recent_volatility']].to_string())
            
            print("\nVolume & Funding Analysis:")
            print(df[['symbol', 'volume_concentration', 'predicted_funding', 'funding_confidence']].to_string())
            
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
            return None

if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    df = analyzer.analyze_markets()
