import asyncio
import json
import os
from datetime import datetime
import pytz
from websockets import connect
from termcolor import cprint
import sys

# Check required packages
print("Checking required packages...")
required_packages = ["websockets", "pytz", "termcolor"]
for package in required_packages:
    if package in sys.modules:
        print(f"✓ {package} is installed")
    else:
        print(f"✗ {package} is missing!")
        sys.exit(1)

# list of symbols i wanna track
symbols = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
    "dogeusdt",
]  # Removed 'wifusdt' as it might not be available in futures
websocket_url = (
    "wss://fstream.binance.com/ws"  # Changed from 'wss://stream.binance.com:9443/ws'
)
trades_filename = "binance_trades.csv"

# check if csv exist and directory is writable
print(f"\nChecking file permissions...")
try:
    if not os.path.exists(trades_filename):
        with open(trades_filename, "w") as f:
            f.write(
                "Event Time, Symbol, Aggregate Trade ID, Price , Quantity, First Trade ID, Trade Time, Is Buyer Maker\n"
            )
        print(f"✓ Created new file: {os.path.abspath(trades_filename)}")
    else:
        print(f"✓ Using existing file: {os.path.abspath(trades_filename)}")
except Exception as e:
    print(f"✗ Error with file access: {str(e)}")
    sys.exit(1)


async def binance_trade_stream(uri, symbol, filename):
    async with connect(uri) as websocket:
        # Subscribe to the stream
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@aggTrade"],
            "id": 1,
        }
        await websocket.send(json.dumps(subscribe_msg))

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                # Skip non-trade messages
                if "e" not in data or data["e"] != "aggTrade":
                    continue

                # Extract trade data
                try:
                    event_time = int(data["E"])
                    agg_trade_id = int(data["a"])
                    price = float(data["p"])
                    quantity = float(data["q"])
                    trade_time = int(data["T"])
                    is_buyer_maker = data["m"]
                except KeyError as e:
                    print(f"Malformed data for {symbol}: {e}")
                    continue

                # Process time and size
                est = pytz.timezone("US/Eastern")
                readable_trade_time = (
                    datetime.fromtimestamp(event_time / 1000)
                    .astimezone(est)
                    .strftime("%H:%M:%S")
                )
                usd_size = price * quantity
                display_symbol = symbol.upper().replace("USDT", "")

                # Process large trades (>$15k)
                if usd_size > 14999:
                    # Determine trade type and base color
                    trade_type = "SELL" if is_buyer_maker else "BUY"
                    color = "red" if trade_type == "SELL" else "green"

                    # Initialize display attributes
                    stars = ""
                    attrs = []

                    # Handle different size tiers
                    if usd_size >= 100000:
                        stars = "*" * 3
                        attrs = ["bold"]
                    elif usd_size >= 50000:
                        stars = "*" * 2
                        attrs = ["bold"]
                        # Adjust colors for large trades
                        if trade_type == "SELL":
                            color = "magenta"
                        else:
                            color = "blue"

                    # Format and display trade
                    output = f"{stars} {trade_type} {display_symbol} {readable_trade_time} ${usd_size:,.0f}"
                    cprint(output, "white", f"on_{color}", attrs=attrs)

                    # Log to CSV
                    with open(filename, "a") as f:
                        f.write(
                            f"{event_time},{symbol.upper()},{agg_trade_id},{price},{quantity},"
                            f"{trade_time},{is_buyer_maker}\n"
                        )

            except asyncio.CancelledError:
                print(f"Shutting down {symbol} stream...")
                break
            except Exception as e:
                print(f"Stream error for {symbol}: {str(e)}")
                await asyncio.sleep(5)


async def main():
    filename = "binance_trades.csv"
    print(f"\nStarting trade stream at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Starting trade stream for symbols: {', '.join(symbol.upper() for symbol in symbols)}"
    )
    print(f"Logging trades to: {os.path.abspath(trades_filename)}")

    try:
        tasks = []
        for symbol in symbols:
            stream_url = f"{websocket_url}/{symbol.lower()}@aggTrade"
            print(f"Connecting to stream: {stream_url}")
            tasks.append(binance_trade_stream(stream_url, symbol, filename))

        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Main loop error: {str(e)}")
    finally:
        print("Trade stream ended")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
