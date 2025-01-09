# common/interfaces/coinbase.py

import ccxt
from typing import Optional, Dict, Any, Tuple
from .base_interface import ExchangeInterface, OrderResponse, PositionInfo

class CoinbaseInterface(ExchangeInterface):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: Optional[str] = None,
        testnet: bool = False
    ):
        super().__init__(api_key, api_secret, testnet)
        
        # Initialize exchange with updated class name
        self.exchange = ccxt.coinbaseadvanced({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # Passphrase from config
            'enableRateLimit': True,
            'urls': {
                'api': base_url or ('https://api-public.sandbox.pro.coinbase.com' if testnet else 'https://api.pro.coinbase.com')
            },
            'options': {
                'adjustForTimeDifference': True
            }
        })
        
        # Set sandbox mode if testnet
        self.exchange.set_sandbox_mode(testnet)
        
        # Load markets to verify connection
        try:
            self.exchange.load_markets()
            self.logger.info("Successfully connected to Coinbase Advanced Trade API.")
        except Exception as e:
            self.logger.error(f"Failed to load markets: {str(e)}")
            raise

    def get_order_book(self, symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        order_book = self.exchange.fetch_order_book(symbol)
        bids = {price: amount for price, amount in order_book['bids']}
        asks = {price: amount for price, amount in order_book['asks']}
        return bids, asks

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_type: str = "limit",
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResponse:
        params = {}
        if reduce_only:
            params['post_only'] = True
        order = self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=size,
            price=price,
            params=params
        )
        return OrderResponse(
            order_id=order['id'],
            symbol=order['symbol'],
            side=order['side'],
            size=order['amount'],
            price=order.get('price', 0.0),
            status=order['status'],
            filled=order.get('filled', 0.0)
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            self.exchange.cancel_all_orders(symbol)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders for {symbol}: {str(e)}")
            return False

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        balance = self.exchange.fetch_balance()
        if symbol in balance['total'] and balance['total'][symbol] > 0:
            # Placeholder implementation; adjust based on actual position data
            return PositionInfo(
                symbol=symbol,
                size=balance['total'][symbol],
                entry_price=0.0,          # Requires actual logic
                leverage=1.0,              # Requires actual logic
                liquidation_price=0.0,      # Requires actual logic
                margin_type="isolated",      # Example value
                unrealized_pnl=0.0,          # Requires actual logic
                side="buy" if balance['total'][symbol] > 0 else "sell"
            )
        return None

    def get_balance(self) -> Dict[str, float]:
        balance = self.exchange.fetch_balance()
        return balance['total']

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        # Coinbase Advanced Trade API does not support leverage via CCXT
        self.logger.warning("Leverage setting is not supported on Coinbase Advanced Trade API via CCXT.")
        return False

    def get_market_price(self, symbol: str) -> Tuple[float, float]:
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['bid'], ticker['ask']

