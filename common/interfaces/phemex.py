from .base_interface import ExchangeInterface, OrderResponse, PositionInfo
import ccxt
from typing import Dict, Any, Optional, Tuple
import logging


class PhemexInterface(ExchangeInterface):
    """
    Interface for interacting with the Phemex exchange using CCXT.
    """

    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        self.exchange = ccxt.phemex({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
            }
        })
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        try:
            order_book = self.exchange.fetch_order_book(symbol)
            self.logger.debug(f"Fetched order book for {symbol}.")
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        params: Dict[str, Any] = {},
    ) -> Optional[OrderResponse]:
        try:
            if order_type.lower() == "limit":
                if price is None:
                    raise ValueError("Price must be specified for limit orders.")
                order = self.exchange.create_limit_order(symbol, side, size, price, params)
            elif order_type.lower() == "market":
                order = self.exchange.create_market_order(symbol, side, size, params)
            else:
                self.logger.error(f"Unsupported order type: {order_type}")
                return None

            order_response = OrderResponse(
                order_id=order.get('id'),
                symbol=symbol,
                side=side,
                size=size,
                price=price if price else 0.0,
                status=order.get('status'),
                filled=order.get('filled', 0.0)
            )
            self.logger.info(f"Placed order: {order_response}")
            return order_response
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Canceled order {order_id} for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            self.exchange.cancel_all_orders(symbol)
            self.logger.info(f"Canceled all orders for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error canceling all orders for {symbol}: {e}")
            return False

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        try:
            positions = self.exchange.fetch_positions()
            for pos in positions:
                if pos['symbol'] == symbol:
                    position_info = PositionInfo(
                        symbol=pos['symbol'],
                        size=pos['contracts'],
                        entry_price=pos['entryPrice'],
                        leverage=pos['leverage'],
                        liquidation_price=pos['liquidationPrice'],
                        margin_type=pos['marginType'],
                        unrealized_pnl=pos['unrealizedPnl'],
                        side=pos['side']
                    )
                    self.logger.debug(f"Fetched position: {position_info}")
                    return position_info
            self.logger.info(f"No open position found for {symbol}.")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
            return None

    def get_balance(self) -> Dict[str, float]:
        try:
            balance = self.exchange.fetch_balance()
            self.logger.debug("Fetched account balance.")
            return balance['total']
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            response = self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"Set leverage to {leverage}x for {symbol}. Response: {response}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting leverage for {symbol}: {e}")
            return False

    def get_market_price(self, symbol: str) -> Tuple[float, float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            bid = ticker.get('bid', 0.0)
            ask = ticker.get('ask', 0.0)
            self.logger.debug(f"Market price for {symbol} - Bid: {bid}, Ask: {ask}")
            return bid, ask
        except Exception as e:
            self.logger.error(f"Error fetching market price for {symbol}: {e}")
            return 0.0, 0.0

    # Implement other necessary methods as needed

