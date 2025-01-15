# backtest_versions/v2/config/training_config.py

CONFIG = {
    "risk": {
        "max_position_size": 1.0,
        "stop_loss_pct": 0.02,
        "trailing_stop_pct": 0.03,
        "vol_lookback": 20,
        "vol_target": 0.15,
        "max_drawdown_limit": 0.20,
        "position_timeout": 5,
        "min_trades_per_month": 5
    },
    
    "rf_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "random_state": 42
    },
    
    "gb_params": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42
    },
    
    "nn_params": {
        "layers": [64, 32, 1],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    
    "ensemble_weights": {
        "rf": 0.4,
        "gb": 0.4,
        "nn": 0.2
    },
    
    "training": {
        "cv_splits": 5,
        "train_size": 0.8,
        "prediction_threshold": 0.7,
        "confidence_threshold": 0.6
    },
    
    "features": {
        "price_features": True,
        "volume_features": True,
        "technical_indicators": True,
        "volatility_features": True,
        "regime_features": True
    }
}
