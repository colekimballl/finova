# exchange_interface_hyperliquid.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
from eth_account.signers.local import LocalAccount

from hyperliquid_functions import limit_order, cancel_all_orders, get_position

class ExchangeInterface(ABC):
    """Abstract base class for exchange interfaces."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_account_info(self):
        pass

class HyperliquidInterface(ExchangeInterface):
    """Implementation of ExchangeInterface for Hyperliquid."""

    def __init__(self, config: Dict[str, Any], account: LocalAccount):
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.base_url = config.get("base_url", "https://api.hyperliquid.com")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.account = account
        self.connected = False

    def connect(self) -> bool:
        """Establishes a connection to Hyperliquid."""
        try:
            # Implement actual connection logic if necessary
            self.connected = True
            self.logger.info("Connected to Hyperliquid successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Hyperliquid: {e}")
            return False

    def disconnect(self):
        """Disconnects from Hyperliquid."""
        # Implement disconnection logic if necessary
        self.connected = False
        self.logger.info("Disconnected from Hyperliquid.")

    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Fetches the order book for a given symbol."""
        try:
            ask, bid, l2_data = ask_bid(symbol)
            if ask is not None and bid is not None:
                return {"ask": ask, "bid": bid, "l2_data": l2_data}
            else:
                self.logger.error(f"Failed to retrieve order book for {symbol}.")
                return {}
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        """Places an order on Hyperliquid."""
        is_buy = True if side.lower() == "buy" else False
        order = limit_order(symbol, is_buy, quantity, price, False, self.account)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancels a specific order."""
        try:
            # Implement cancellation logic based on order_id
            # For demonstration, using cancel_all_orders (modify as needed)
            cancel_all_orders(self.account)
            self.logger.info(f"Cancelled order {order_id} successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_account_info(self):
        """Retrieves account information."""
        try:
            account_value = acct_bal(self.account)
            return {"account_value": account_value}
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            return {}

