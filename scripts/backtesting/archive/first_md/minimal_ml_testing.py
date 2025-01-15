# minimal_ml_testing.py
import warnings
import logging
from ml_config import ML_CONFIG
from feature_generator import FeatureGenerator
import lightgbm as lgb
import pandas as pd
import numpy as np

# Suppress specific FutureWarnings
warnings.filterwarnings(
    'ignore',
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize FeatureGenerator
    feature_gen = FeatureGenerator(ML_CONFIG)
    
    # Load data
    logger.info("Loading data...")
    data = pd.read_csv('data/stocks_daily.csv', parse_dates=['date'])
    
    # Ensure necessary columns
    required_columns = {'symbol', 'open', 'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        logger.error(f"Missing columns in data: {missing}")
        return
    
    # Set 'date' as index
    data.set_index('date', inplace=True)
    
    # Create target
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data = data[:-1]  # Drop last row with NaN target
    
    # Generate features
    logger.info("Generating features...")
    features = feature_gen.generate_features(data)
    
    # Clean features
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.ffill(inplace=True)
    features.fillna(0, inplace=True)
    
    # Select a single symbol for testing
    symbol = 'AAPL'
    symbol_data = data[data['symbol'] == symbol]
    symbol_features = features.xs(symbol, level='symbol')
    
    # Split into train and test
    train_size = int(len(symbol_data) * ML_CONFIG['training']['train_size'])
    train_features = symbol_features.iloc[:train_size]
    train_data = symbol_data.iloc[:train_size]
    test_features = symbol_features.iloc[train_size:]
    test_data = symbol_data.iloc[train_size:]
    
    # Train model
    logger.info("Training model...")
    model = lgb.LGBMClassifier(**ML_CONFIG['model']['params'])
    model.fit(train_features, train_data['target'])
    logger.info("Model training completed.")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict_proba(test_features)[:, 1]
    
    # Generate signals
    signals = pd.Series(index=test_data.index, dtype='object')
    signals[predictions > 0.7] = 'buy'
    signals[predictions < 0.3] = 'sell'
    
    # Calculate returns
    returns = test_data['close'].pct_change().shift(-1)
    strategy_returns = returns * signals.map({'buy': 1, 'sell': -1})
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    
    # Log results
    total_return = cumulative_returns.iloc[-1] - 1
    logger.info(f"Total Return for {symbol}: {total_return:.2%}")

if __name__ == "__main__":
    main()

