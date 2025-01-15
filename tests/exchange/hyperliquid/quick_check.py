import os
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
from dotenv import load_dotenv
from termcolor import cprint

def print_account_info():
    try:
        # Load environment
        load_dotenv()
        
        # Your actual trading address
        TRADING_ADDRESS = "0x21df77DB5bb9670eCeA0aF35EeF838E3391fbB48"
        
        # Get private key - using API_SECRET for authentication
        private_key = os.getenv("HYPERLIQUID_API_SECRET")
        if private_key.startswith('0x'):
            private_key = private_key[2:]

        # Setup
        auth_account = Account.from_key(private_key)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get info for the trading address
        user_state = info.user_state(TRADING_ADDRESS)
        
        cprint("\n🏦 Account Overview", "cyan", attrs=["bold"])
        print(f"trading address: {TRADING_ADDRESS[:10]}...{TRADING_ADDRESS[-6:]}")
        print(f"auth address: {auth_account.address[:10]}...{auth_account.address[-6:]}")
        
        # Account value
        margin_summary = user_state.get('marginSummary', {})
        account_value = float(margin_summary.get('accountValue', 0))
        print(f"\nbalance: ${account_value:.2f}")
        print(f"total margin: ${float(margin_summary.get('totalMargin', 0)):.2f}")
        print(f"unrealized pnl: ${float(margin_summary.get('unrealizedPnl', 0)):.2f}")

        # Positions check
        positions = user_state.get("assetPositions", [])
        if positions:
            cprint("\n📊 Open Positions", "green")
            for pos in positions:
                p = pos["position"]
                print(f"symbol: {p['coin']}")
                print(f"size  : {p['szi']}")
                print(f"pnl   : {float(p.get('returnOnEquity', 0)) * 100:.2f}%")
        else:
            cprint("\nno open positions", "yellow")

        # Orders check
        orders = info.open_orders(TRADING_ADDRESS)
        if orders:
            cprint("\n📋 Pending Orders", "blue")
            for order in orders:
                print(f"symbol: {order['coin']}")
                print(f"size  : {order['sz']}")
                print(f"price : ${float(order['px']):.2f}")
        else:
            cprint("\nno pending orders", "yellow")

    except Exception as e:
        cprint(f"\n❌ Error: {str(e)}", "red")
        cprint("\nDebug Info:", "yellow")
        print(f"API_SECRET starts with: {private_key[:6]}...")

if __name__ == "__main__":
    print_account_info()
