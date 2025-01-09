from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

@dataclass
class OrderResponse:
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    status: str
    filled: float = 0.0

@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_price: float
    leverage: float
    liquidation_price: float
    margin_type: str
    unrealized_pnl: float
    side: str

class ExchangeInterface(ABC):
    """Base class for all exchange interfaces"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_order_book(self, symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get current order book"""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResponse:
        """Place an order"""
        pass

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order"""
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all orders for a symbol"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get current position information"""
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        pass

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        pass

    @abstractmethod
    def get_market_price(self, symbol: str) -> Tuple[float, float]:
        """Get current market price (returns tuple of bid, ask)"""
        pass

    def calculate_position_value(self, position: PositionInfo) -> float:
        """Calculate USD value of position"""
        mark_price = self.get_market_price(position.symbol)[0]  # Using bid price
        return abs(position.size * mark_price)

    def calculate_liquidation_risk(self, position: PositionInfo) -> float:
        """Calculate distance to liquidation as percentage"""
        mark_price = self.get_market_price(position.symbol)[0]
        return abs((mark_price - position.liquidation_price) / mark_price) * 100

    def is_position_safe(self, position: PositionInfo, min_liq_distance: float = 15.0) -> bool:
        """Check if position has safe distance to liquidation"""
        liq_distance = self.calculate_liquidation_risk(position)
        return liq_distance >= min_liq_distance
