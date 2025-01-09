# exchange_interface_phemex.py

import ccxt
from typing import Any, Dict, Optional
from logger import setup_logger

logger = setup_logger("exchange_interface_phemex")

class PhemexClient:
    """
    Interface for interacting with the Phemex exchange using CCXT.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.phemex.com"):
        """
        Initializes the Phemex client with API credentials.

        :param api_key: Phemex API key.
        :param api_secret: Phemex API secret.
        :param base_url: Base URL for Phemex API.
        """
        self.exchange = ccxt.phemex({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'urls': {
                'api': base_url
            },
            'options': {
                'defaultType': 'swap',
            }
        })
        self.logger = logger
        self.connected = False

    def connect(self) -> bool:
        """
        Establishes a connection to Phemex by loading markets.

        :return: True if connected successfully, False otherwise.
        """
        try:
            self.exchange.load_markets()
            self.connected = True
            self.logger.info("Connected to Phemex successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Phemex: {e}")
            return False

    def disconnect(self):
        """
        Disconnects from Phemex. (CCXT does not require explicit disconnection.)
        """
        self.connected = False
        self.logger.info("Disconnected from Phemex.")

    def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: Optional[float] = None, params: Dict[str, Any] = {}) -> Optional[Dict[str, Any]]:
        """
        Places an order on Phemex.

        :param symbol: Trading pair (e.g., 'BTC/USD').
        :param side: 'buy' or 'sell'.
        :param order_type: 'limit' or 'market'.
        :param amount: Amount to trade.
        :param price: Limit price. Required for limit orders.
        :param params: Additional parameters.
        :return: Order details if successful, None otherwise.
        """
        try:
            if order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price must be set for limit orders.")
                if side.lower() == 'buy':
                    order = self.exchange.create_limit_buy_order(symbol, amount, price, params)
                else:
                    order = self.exchange.create_limit_sell_order(symbol, amount, price, params)
            elif order_type.lower() == 'market':
                if side.lower() == 'buy':
                    order = self.exchange.create_market_buy_order(symbol, amount, params)
                else:
                    order = self.exchange.create_market_sell_order(symbol, amount, params)
            else:
                self.logger.error(f"Unsupported order type: {order_type}")
                return None
            self.logger.info(f"Placed {side} {order_type} order: {order}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing {side} {order_type} order for {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancels a specific order.

        :param symbol: Trading pair.
        :param order_id: ID of the order to cancel.
        :return: True if canceled successfully, False otherwise.
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancels all open orders for a specific symbol.

        :param symbol: Trading pair.
        :return: True if all orders are canceled successfully, False otherwise.
        """
        try:
            self.exchange.cancel_all_orders(symbol)
            self.logger.info(f"Cancelled all orders for {symbol}.")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling all orders for {symbol}: {e}")
            return False

    def fetch_open_positions(self) -> Optional[list]:
        """
        Fetches all open positions.

        :return: List of open positions if successful, None otherwise.
        """
        try:
            positions = self.exchange.fetch_positions()
            self.logger.info(f"Fetched open positions: {positions}")
            return positions
        except Exception as e:
            self.logger.error(f"Error fetching open positions: {e}")
            return None

    def fetch_balance(self, params: Dict[str, Any] = {}) -> Optional[Dict[str, Any]]:
        """
        Fetches the account balance.

        :param params: Additional parameters.
        :return: Balance information if successful, None otherwise.
        """
        try:
            balance = self.exchange.fetch_balance(params=params)
            self.logger.info(f"Fetched balance: {balance}")
            return balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return None

    def fetch_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the order book for a given symbol.

        :param symbol: Trading pair.
        :return: Order book data if successful, None otherwise.
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol)
            self.logger.info(f"Fetched order book for {symbol}.")
            return order_book
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

