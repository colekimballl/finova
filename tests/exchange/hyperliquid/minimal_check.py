import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

def fetch_trades(symbol, api_key=None, lookback_hours=72, limit=100):
    """
    Fetches historical trade data for a given symbol from Hyperliquid API.

    Parameters:
        symbol (str): The trading symbol (e.g., 'BTCUSD').
        api_key (str): Your API key if authentication is required.
        lookback_hours (int): The number of hours to look back for trades.
        limit (int): The maximum number of trades to fetch.

    Returns:
        pd.DataFrame or None: DataFrame containing trade data or None if failed.
    """
    # Confirm the correct endpoint
    url = "https://api.hyperliquid.xyz/info"  # Replace with the correct endpoint if different

    # Set headers
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Define multiple payload variants to try
    payload_variants = [
        {
            "type": "trades",
            "symbol": symbol.upper(),
            "limit": limit,
            "start_time": int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000)
        },
        {
            "type": "trades",
            "coin": symbol.upper(),
            "limit": limit,
            "start_time": int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000)
        },
        {
            "type": "trades",
            "symbol": symbol.upper(),
            "limit": limit  # Without 'start_time'
        },
        {
            "symbol": symbol.upper(),
            "limit": limit  # Without 'type'
        },
    ]

    # List of HTTP methods to try
    methods = ['POST', 'GET']

    for method in methods:
        for payload in payload_variants:
            print(f"=== {method} Request ===")
            print(f"Payload/Params: {json.dumps(payload, indent=2)}")
            try:
                if method == 'POST':
                    response = requests.post(url, headers=headers, json=payload)
                elif method == 'GET':
                    response = requests.get(url, headers=headers, params=payload)
                else:
                    continue

                print(f"Status Code: {response.status_code}")
                try:
                    response_json = response.json()
                    print(f"Response JSON:")
                    print(json.dumps(response_json, indent=2))
                    
                    # Check if response contains trade data
                    if response.status_code == 200 and isinstance(response_json, list) and response_json:
                        df = pd.DataFrame(response_json)
                        required_columns = {'px', 'sz', 'time'}
                        if required_columns.issubset(df.columns):
                            # Convert data types
                            df['px'] = pd.to_numeric(df['px'], errors='coerce')
                            df['sz'] = pd.to_numeric(df['sz'], errors='coerce')
                            df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                            df.dropna(subset=['px', 'sz', 'time'], inplace=True)
                            df = df.sort_values('time').reset_index(drop=True)
                            print(f"Fetched {len(df)} trades for {symbol}.")
                            return df
                        else:
                            print(f"Missing required columns. Available columns: {df.columns.tolist()}")
                    else:
                        print("No trade data available or unsuccessful request.")
                except json.JSONDecodeError:
                    print(f"Response Text: {response.text}")

            except Exception as e:
                print(f"An exception occurred: {e}")
            print("\n")

    print("All payload variants and methods failed to fetch trade data.")
    return None

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve API key from environment variable
    api_key = os.getenv("HYPERLIQUID_API_KEY")  # Set this in your .env file or environment
    if not api_key:
        print("No API key found. Proceeding without authentication.")

    # Define the trading symbol
    symbol = "BTC"  # Adjust as needed, e.g., "BTCUSD"

    # Fetch trade data
    trades_df = fetch_trades(symbol, api_key=api_key, lookback_hours=72, limit=100)

    if trades_df is not None:
        # Display sample data
        print("\n=== Sample Trade Data ===")
        print(trades_df.head())

        # Save to CSV
        csv_filename = f"trades_{symbol.upper()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(csv_filename, index=False)
        print(f"\nTrade data saved to {csv_filename}")
    else:
        print("\nFailed to fetch trade data.")

if __name__ == "__main__":
    main()

