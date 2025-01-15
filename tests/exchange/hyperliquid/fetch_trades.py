import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from eth_account import Account
from eth_account.messages import encode_defunct
import hashlib
import hmac

def load_credentials():
    """
    Load API credentials from environment variables.
    """
    load_dotenv()
    wallet = os.getenv("HYPERLIQUID_API_KEY")
    secret = os.getenv("HYPERLIQUID_API_SECRET")
    if not wallet or not secret:
        raise ValueError("API credentials not found in environment variables.")
    return wallet, secret

def sign_payload(payload, secret):
    """
    Sign the payload using HMAC-SHA256 with the secret key.

    Parameters:
        payload (dict): The JSON payload to sign.
        secret (str): The secret key for signing.

    Returns:
        str: The hexadecimal signature.
    """
    # Serialize the payload to a JSON-formatted string
    payload_string = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    # Create HMAC-SHA256 signature
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def fetch_trades(symbol, wallet, secret, lookback_hours=72, limit=100):
    """
    Fetches historical trade data for a given symbol from Hyperliquid API.

    Parameters:
        symbol (str): The trading symbol (e.g., 'BTCUSD').
        wallet (str): The wallet address.
        secret (str): The secret key for signing.
        lookback_hours (int): The number of hours to look back for trades.
        limit (int): The maximum number of trades to fetch.

    Returns:
        pd.DataFrame or None: DataFrame containing trade data or None if failed.
    """
    # Define the correct endpoint for trades
    url = "https://api.hyperliquid.xyz/trades"  # Replace with the correct endpoint if different

    # Prepare the payload
    payload = {
        "symbol": symbol.upper(),
        "limit": limit,
        "start_time": int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000),
        "wallet": wallet
    }

    # Sign the payload
    signature = sign_payload(payload, secret)
    payload["signature"] = signature

    # Set headers
    headers = {
        "Content-Type": "application/json",
    }

    try:
        # Send the POST request
        response = requests.post(url, headers=headers, json=payload)
        print(f"=== {symbol} Trades Request ===")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            trades = response.json()
            if isinstance(trades, list) and trades:
                df = pd.DataFrame(trades)
                required_columns = {'px', 'sz', 'time'}
                if not required_columns.issubset(df.columns):
                    print(f"Missing required columns: {df.columns.tolist()}")
                    return None
                # Convert data types
                df['px'] = pd.to_numeric(df['px'], errors='coerce')
                df['sz'] = pd.to_numeric(df['sz'], errors='coerce')
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                df.dropna(subset=['px', 'sz', 'time'], inplace=True)
                df = df.sort_values('time').reset_index(drop=True)
                print(f"Fetched {len(df)} trades for {symbol} within the last {lookback_hours} hours.")
                return df
            else:
                print("No trade data available.")
                return None
        else:
            try:
                error_info = response.json()
                print(f"Error Response: {json.dumps(error_info, indent=2)}")
            except json.JSONDecodeError:
                print(f"Error Response Text: {response.text}")
            return None

    except Exception as e:
        print(f"An exception occurred: {e}")
        return None

def main():
    try:
        # Load API credentials
        wallet, secret = load_credentials()
    except ValueError as ve:
        print(ve)
        return

    # Define the trading symbol
    symbol = "BTCUSD"  # Adjust as per API requirements (e.g., "BTCUSD", "ETHUSD")

    # Fetch trade data
    trades_df = fetch_trades(symbol, wallet, secret, lookback_hours=72, limit=100)

    if trades_df is not None:
        # Display sample data
        print("\n=== Sample Trade Data ===")
        print(trades_df.head())

        # Save to CSV
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"trades_{symbol.upper()}_{timestamp}.csv"
        trades_df.to_csv(csv_filename, index=False)
        print(f"\nTrade data saved to {csv_filename}")
    else:
        print("\nFailed to fetch trade data.")

if __name__ == "__main__":
    main()

