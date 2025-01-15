import numpy as np
from typing import List, Dict

class ProfitOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.position_sizes = []
        self.entry_points = []
        self.exit_points = []
        
    def calculate_optimal_size(self, 
                             conviction: float,
                             current_price: float,
                             volatility: float,
                             available_capital: float) -> float:
        """Calculate optimal position size based on multiple factors"""
        base_size = np.random.uniform(
            self.config['portfolio']['max_position_size'][0],
            self.config['portfolio']['max_position_size'][1]
        )
        
        # Adjust size based on conviction
        conviction_modifier = conviction * 1.5
        
        # Adjust size based on volatility
        vol_modifier = 1 - (volatility * 0.5)  # Reduce size in high volatility
        
        # Calculate final size
        optimal_size = base_size * conviction_modifier * vol_modifier
        
        # Ensure we don't exceed available capital
        position_value = optimal_size * available_capital
        max_position = available_capital * self.config['portfolio']['max_position_size'][1]
        
        if position_value > max_position:
            optimal_size = max_position / available_capital
            
        return optimal_size
        
    def calculate_exit_points(self, 
                            entry_price: float, 
                            position_size: float,
                            prediction_confidence: float) -> List[Dict]:
        """Calculate multiple exit points for profit taking"""
        exit_points = []
        
        # Base profit targets on prediction confidence
        profit_targets = [level * (1 + prediction_confidence) 
                         for level in self.config['portfolio']['profit_taking_levels']]
        
        # Create scaled exit points
        position_remaining = position_size
        for i, target in enumerate(profit_targets):
            exit_size = position_size * (1 / (len(profit_targets) - i))
            exit_points.append({
                'price': entry_price * (1 + target),
                'size': min(exit_size, position_remaining)
            })
            position_remaining -= exit_size
            
        return exit_points
