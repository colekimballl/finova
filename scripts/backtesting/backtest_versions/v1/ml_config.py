# ml_config.py
ML_CONFIG = {
    'portfolio': {
        'initial_capital': 10000,
        'max_position_size': (0.05, 0.99),
        'transaction_cost': 0.001,
        'target_profit': 0.02,  # 2% target profit per trade
        'max_drawdown': 0.15,   # 15% maximum drawdown
        'profit_taking_levels': [0.02, 0.05, 0.10],  # Multiple profit-taking levels
        'position_scaling': True  # Scale positions based on conviction
    },
    
    'optimization': {
        'objective': 'sharpe',  # Could be 'profit', 'sharpe', or 'sortino'
        'lookback_windows': [5, 10, 20, 50],  # Dynamic timeframes
        'profit_weight': 0.7,  # Weight for profit in scoring
        'risk_weight': 0.3,    # Weight for risk in scoring
        'min_trades': 20,      # Minimum trades for validation
    },
    
    'features': {
        'price': ['returns', 'log_returns'],
        'momentum': [5, 10, 20, 50, 200],
        'volatility': [5, 10, 20],
        'rsi': [7, 14, 21],
        'bollinger': [20],
        'macd': [(12, 26, 9)],
        'custom': [
            'price_trend_strength',
            'volume_trend_strength',
            'profit_factor',
            'win_rate'
        ]
    },

    'training': {
        'train_size': 0.8,       # 80% of data for training
        'test_size': 0.2,        # 20% of data for testing
        'random_state': 42       # Seed for reproducibility
    },

    'model': {
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'boosting_type': 'gbdt',
            'objective': 'binary',  # Assuming binary classification
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': 42
            # Add other LightGBM parameters as needed
        }
    }
}

