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
symbols = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "dogeusdt"]
websocket_url = "wss://fstream.binance.com/ws"
trades_filename = "binance_trades.csv"

# check if csv exist and directory is writable
print(f"\nChecking file permissions...")
try:
    if not os.path.exists(trades_filename):
        with open(trades_filename, "w") as f:
            f.write(
                "Event Time, Symbol, Aggregate Trade ID, Price, Quantity, Trade Time, Is Buyer Maker, Market Share\n"
            )
        print(f"✓ Created new file: {os.path.abspath(trades_filename)}")
    else:
        print(f"✓ Using existing file: {os.path.abspath(trades_filename)}")
except Exception as e:
    print(f"✗ Error with file access: {str(e)}")
    sys.exit(1)


class TradeAggregator:
    def __init__(self):
        self.trade_buckets = {}
        self.total_volume = {}  # Track volume per symbol
        self.daily_stats = {
            "largest_trade": {"size": 0, "symbol": None, "time": None, "type": None},
            "total_large_trades": 0,
            "volume_by_symbol": {},
            "large_trades_by_symbol": {},
        }

    async def add_trade(self, symbol, second, usd_size, is_buyer_maker):
        # Update volume tracking
        self.total_volume[symbol] = self.total_volume.get(symbol, 0) + usd_size

        # Update daily stats
        self.daily_stats["volume_by_symbol"][symbol] = (
            self.daily_stats["volume_by_symbol"].get(symbol, 0) + usd_size
        )

        if usd_size > 500000:  # Only track large trades
            trade_key = (symbol, second, is_buyer_maker)
            self.trade_buckets[trade_key] = (
                self.trade_buckets.get(trade_key, 0) + usd_size
            )

            # Update largest trade if current trade is bigger
            if usd_size > self.daily_stats["largest_trade"]["size"]:
                self.daily_stats["largest_trade"] = {
                    "size": usd_size,
                    "symbol": symbol,
                    "time": second,
                    "type": "SELL" if is_buyer_maker else "BUY",
                }

            self.daily_stats["total_large_trades"] += 1
            self.daily_stats["large_trades_by_symbol"][symbol] = (
                self.daily_stats["large_trades_by_symbol"].get(symbol, 0) + 1
            )

    async def print_daily_summary(self):
        cprint("\n=== Daily Statistics ===", "yellow", attrs=["bold"])
        largest = self.daily_stats["largest_trade"]
        if largest["size"] > 0:
            cprint(
                f"Largest Trade: {largest['type']} {largest['symbol']} ${largest['size']/1000000:.2f}M at {largest['time']}",
                "yellow",
                attrs=["bold"],
            )

        cprint(
            f"Total Large Trades (>$500K): {self.daily_stats['total_large_trades']}",
            "yellow",
        )

        if self.daily_stats["volume_by_symbol"]:
            cprint("\nVolume by Symbol:", "yellow")
            for symbol, volume in self.daily_stats["volume_by_symbol"].items():
                large_trades = self.daily_stats["large_trades_by_symbol"].get(symbol, 0)
                cprint(
                    f"{symbol}: ${volume/1000000:.2f}M ({large_trades} large trades)",
                    "yellow",
                )

    async def check_and_print_trades(self):
        time_stamp_now = datetime.utcnow().strftime("%H:%M:%S")
        deletions = []

        for trade_key, usd_size in self.trade_buckets.items():
            symbol, second, is_buyer_maker = trade_key
            if second < time_stamp_now and usd_size > 500000:
                attrs = ["bold"]
                back_color = "on_blue" if not is_buyer_maker else "on_magenta"
                trade_type = "BUY" if not is_buyer_maker else "SELL"

                # Calculate market share
                market_share = (
                    usd_size / self.total_volume.get(symbol, usd_size)
                ) * 100

                # Add blinking for massive trades
                if usd_size > 5000000:
                    attrs.append("blink")

                # Format size in millions
                size_in_millions = usd_size / 1000000

                # Different formatting based on size
                if usd_size > 3000000:
                    trade_str = f"{trade_type} {symbol} {second} ${size_in_millions:.2f}M ({market_share:.1f}% vol)"
                    cprint(trade_str, "white", back_color, attrs=attrs)
                else:
                    trade_str = f"{trade_type} {symbol} {second} ${size_in_millions:.2f}M ({market_share:.1f}% vol)"
                    cprint(trade_str, "white", back_color, attrs=attrs)

                deletions.append(trade_key)

        for key in deletions:
            del self.trade_buckets[key]


trade_aggregator = TradeAggregator()


async def binance_trade_stream(uri, symbol, filename, aggregator):
    async with connect(uri) as websocket:
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

                try:
                    price = float(data["p"])
                    quantity = float(data["q"])
                    usd_size = price * quantity
                    trade_time = datetime.fromtimestamp(
                        int(data["T"]) / 1000, pytz.timezone("US/Eastern")
                    )
                    readable_trade_time = trade_time.strftime("%H:%M:%S")

                    await aggregator.add_trade(
                        symbol.upper().replace("USDT", ""),
                        readable_trade_time,
                        usd_size,
                        data["m"],
                    )

                except KeyError as e:
                    print(f"Malformed data for {symbol}: {e}")
                    continue

            except Exception as e:
                print(f"Stream error for {symbol}: {str(e)}")
                await asyncio.sleep(5)


async def print_aggregated_trades_every_second(aggregator):
    summary_counter = 0
    while True:
        await asyncio.sleep(1)
        await aggregator.check_and_print_trades()

        # Print summary every 5 minutes (300 seconds)
        summary_counter += 1
        if summary_counter >= 300:
            await aggregator.print_daily_summary()
            summary_counter = 0


async def main():
    print(f"\nStarting trade stream at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Starting trade stream for symbols: {', '.join(symbol.upper() for symbol in symbols)}"
    )
    print("Trade summary will be printed every 5 minutes")

    try:
        trade_stream_tasks = []
        for symbol in symbols:
            stream_url = f"{websocket_url}/{symbol.lower()}@aggTrade"
            print(f"Connecting to stream: {stream_url}")
            trade_stream_tasks.append(
                binance_trade_stream(
                    stream_url, symbol, trades_filename, trade_aggregator
                )
            )

        print_task = asyncio.create_task(
            print_aggregated_trades_every_second(trade_aggregator)
        )
        await asyncio.gather(*trade_stream_tasks, print_task)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        await trade_aggregator.print_daily_summary()
    except Exception as e:
        print(f"Main loop error: {str(e)}")
    finally:
        print("Trade stream ended")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
