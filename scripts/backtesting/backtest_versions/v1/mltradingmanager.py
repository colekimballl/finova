import numpy as np 
from typing import Dict, List
from feature_generator import FeatureGenerator
from profit_optimizer import ProfitOptimizer


class MLTradingManager:
    def __init__(self, config: dict):
        self.config = config
        self.optimizer = ProfitOptimizer(config)
        self.feature_generator = FeatureGenerator(config)
        self.current_positions = {}
        self.trade_history = []
        self.portfolio_value = config['portfolio']['initial_capital']
        
    def execute_trade(self, 
                     symbol: str, 
                     prediction: float,
                     confidence: float,
                     current_price: float,
                     volatility: float):
        """Execute trade with profit optimization"""
        # Calculate position size
        available_capital = self.portfolio_value * (1 - sum(self.current_positions.values()))
        size = self.optimizer.calculate_optimal_size(
            conviction=confidence,
            current_price=current_price,
            volatility=volatility,
            available_capital=available_capital
        )
        
        # Calculate exit points
        exit_points = self.optimizer.calculate_exit_points(
            entry_price=current_price,
            position_size=size,
            prediction_confidence=confidence
        )
        
        # Execute trade
        cost = size * current_price * self.config['portfolio']['transaction_cost']
        self.portfolio_value -= cost
        
        # Record position
        self.current_positions[symbol] = size
        
        return {
            'symbol': symbol,
            'entry_price': current_price,
            'size': size,
            'exit_points': exit_points,
            'cost': cost
        }
