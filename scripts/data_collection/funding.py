import asyncio
import json
from datetime import datetime, timedelta
from websockets import connect
from termcolor import cprint
import sys
import itertools
import time
import aiohttp

# Configuration
symbols = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "dogeusdt"]
base_websocket_url = "wss://fstream.binance.com/ws"
rest_api_url = "https://fapi.binance.com/fapi/v1"
debug_mode = False


class Spinner:
    def __init__(self):
        self.spinner = itertools.cycle(
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        )
        self.start_time = time.time()

    def spin(self, status):
        elapsed_time = time.time() - self.start_time
        return f"\r{next(self.spinner)} {status} ({elapsed_time:.1f}s)"


class MarketData:
    def __init__(self):
        self.funding_rates = {}
        self.mark_prices = {}
        self.last_prices = {}
        self.next_funding_times = {}
        self.high_rates = {symbol: -float("inf") for symbol in symbols}
        self.low_rates = {symbol: float("inf") for symbol in symbols}
        self.significant_changes = []
        self.last_update = {}

    def update_data(self, symbol, rate, mark_price, next_funding_time):
        old_rate = self.funding_rates.get(symbol)

        self.funding_rates[symbol] = rate
        self.mark_prices[symbol] = mark_price
        self.next_funding_times[symbol] = next_funding_time
        self.last_update[symbol] = datetime.now()

        if rate > self.high_rates[symbol]:
            self.high_rates[symbol] = rate
        if rate < self.low_rates[symbol]:
            self.low_rates[symbol] = rate

        if old_rate is not None and abs(rate - old_rate) > 0.005:
            self.significant_changes.append(
                {
                    "symbol": symbol,
                    "time": datetime.now(),
                    "old_rate": old_rate,
                    "new_rate": rate,
                    "change": rate - old_rate,
                }
            )

    def get_summary(self):
        if not self.funding_rates:
            return "No data yet"

        return {
            "highest": max(self.funding_rates.items(), key=lambda x: x[1]),
            "lowest": min(self.funding_rates.items(), key=lambda x: x[1]),
            "average": sum(self.funding_rates.values()) / len(self.funding_rates),
        }


class OutputManager:
    def __init__(self):
        self.last_summary_time = datetime.now()
        self.summary_interval = timedelta(minutes=5)

    def get_color_scheme(self, rate):
        if rate > 0.1:
            return ("white", "on_red", "🔥")
        elif rate > 0.05:
            return ("black", "on_yellow", "⚠️")
        elif rate > 0.01:
            return ("white", "on_cyan", "📈")
        elif rate > -0.01:
            return ("white", "on_green", "✅")
        else:
            return ("white", "on_blue", "💰")

    def format_next_funding(self, next_funding_time):
        time_until = next_funding_time - datetime.now()
        hours, remainder = divmod(time_until.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}"

    async def print_rate(self, symbol, rate, mark_price, next_funding_time):
        text_color, back_color, indicator = self.get_color_scheme(rate)
        time_until = self.format_next_funding(next_funding_time)
        current_time = datetime.now().strftime("%H:%M:%S")

        annual_rate = rate * 3 * 365 * 100

        output = (
            f"[{current_time}] {indicator} {symbol:<5} | "
            f"Rate: {rate:>7.4f}% | "
            f"Annual: {annual_rate:>7.2f}% | "
            f"Price: ${mark_price:>10,.2f} | "
            f"Next Funding: {time_until}"
        )
        cprint(output, text_color, back_color)
        sys.stdout.flush()


market_data = MarketData()
output_manager = OutputManager()
spinner = Spinner()


async def get_initial_data(symbol):
    try:
        # Get funding rate
        funding_url = f"{rest_api_url}/fundingRate?symbol={symbol.upper()}&limit=1"
        price_url = f"{rest_api_url}/ticker/price?symbol={symbol.upper()}"
        next_funding_url = f"{rest_api_url}/premiumIndex?symbol={symbol.upper()}"

        async with aiohttp.ClientSession() as session:
            # Get funding rate
            async with session.get(funding_url) as response:
                funding_data = await response.json()
                if not funding_data:
                    raise ValueError("No funding data received")
                funding_rate = float(funding_data[0]["fundingRate"]) * 100

            # Get current price
            async with session.get(price_url) as response:
                price_data = await response.json()
                mark_price = float(price_data["price"])

            # Get next funding time
            async with session.get(next_funding_url) as response:
                next_funding_data = await response.json()
                next_funding = datetime.fromtimestamp(
                    int(next_funding_data["nextFundingTime"]) / 1000
                )

            return funding_rate, mark_price, next_funding
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")


async def binance_stream(symbol):
    try:
        print(f"Fetching initial data for {symbol}...")
        funding_rate, mark_price, next_funding = await get_initial_data(symbol)
        symbol_display = symbol.upper().replace("USDT", "")

        market_data.update_data(symbol_display, funding_rate, mark_price, next_funding)
        await output_manager.print_rate(
            symbol_display, funding_rate, mark_price, next_funding
        )
    except Exception as e:
        print(f"Error getting initial data for {symbol}: {e}")

    stream_url = f"{base_websocket_url}/{symbol.lower()}@markPrice"
    sys.stdout.write(spinner.spin(f"Starting stream for {symbol}"))
    sys.stdout.flush()

    async with connect(stream_url) as websocket:
        sys.stdout.write(f"\r✓ {symbol} stream connected" + " " * 50 + "\n")
        sys.stdout.flush()

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)

                if "e" not in data or data["e"] != "markPriceUpdate":
                    continue

                symbol_display = data["s"].replace("USDT", "")
                funding_rate = float(data["r"]) * 100
                mark_price = float(data["p"])
                next_funding = datetime.fromtimestamp(int(data["T"]) / 1000)

                market_data.update_data(
                    symbol_display, funding_rate, mark_price, next_funding
                )
                await output_manager.print_rate(
                    symbol_display, funding_rate, mark_price, next_funding
                )

            except Exception as e:
                if debug_mode:
                    print(f"Error in {symbol} stream: {str(e)}")
                await asyncio.sleep(5)


async def print_periodic_summary():
    while True:
        await asyncio.sleep(300)
        summary = market_data.get_summary()
        if isinstance(summary, dict):
            cprint("\n=== Funding Rate Summary ===", "white", "on_blue", attrs=["bold"])
            cprint(
                f"Highest: {summary['highest'][0]} @ {summary['highest'][1]:.4f}%",
                "white",
                "on_blue",
            )
            cprint(
                f"Lowest: {summary['lowest'][0]} @ {summary['lowest'][1]:.4f}%",
                "white",
                "on_blue",
            )
            cprint(f"Average Rate: {summary['average']:.4f}%", "white", "on_blue")

            if market_data.significant_changes:
                recent_changes = market_data.significant_changes[-3:]
                cprint("\nRecent Significant Changes:", "white", "on_red")
                for change in recent_changes:
                    cprint(
                        f"{change['symbol']}: {change['old_rate']:.4f}% → {change['new_rate']:.4f}% "
                        f"(Δ {change['change']:.4f}%)",
                        "white",
                        "on_red",
                    )
            print("-" * 80)


async def main():
    print("\n=== Funding Rate Monitor ===")
    print(f"Monitoring: {', '.join(symbol.upper() for symbol in symbols)}")
    print("\nColor Coding (8-hour rates):")
    cprint("🔥 > 0.10% (Extreme)", "white", "on_red")
    cprint("⚠️ > 0.05% (High)", "black", "on_yellow")
    cprint("📈 > 0.01% (Notable)", "white", "on_cyan")
    cprint("✅ > -0.01% (Normal)", "white", "on_green")
    cprint("💰 < -0.01% (Low)", "white", "on_blue")
    print("\nInitializing...\n")

    try:
        tasks = [binance_stream(symbol) for symbol in symbols]
        tasks.append(print_periodic_summary())
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        if debug_mode:
            print(f"Main loop error: {str(e)}")
    finally:
        print("\nFinal Summary:")
        summary = market_data.get_summary()
        if isinstance(summary, dict):
            print(
                f"Highest Rate: {summary['highest'][0]} @ {summary['highest'][1]:.4f}%"
            )
            print(f"Lowest Rate: {summary['lowest'][0]} @ {summary['lowest'][1]:.4f}%")
        print("\nMonitor ended")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
