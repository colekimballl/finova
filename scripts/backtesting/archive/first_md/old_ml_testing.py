# ml_testing.py
import logging
import warnings
from ml_config import ML_CONFIG
from feature_generator import FeatureGenerator
from data_scripts.data_loader import MarketDataLoader
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self, config: Dict):
        """Initialize backtest with configuration"""
        self.config = config
        self.feature_generator = FeatureGenerator(config)
        self.data_loader = MarketDataLoader()
        self.results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")

    def run_backtest(self, market: str, timeframe: str) -> None:
        """Run backtest for a specific market and timeframe"""
        try:
            logger.info(f"Starting backtest for {market} {timeframe}")
            
            # Load and prepare data
            data = self._prepare_data(market, timeframe)
            if data is None:
                return
                
            # Generate features
            features = self.feature_generator.generate_features(data)
            if features is None:
                return
            
            # Run backtest for each symbol
            results = {}
            for symbol in data['symbol'].unique():
                logger.info(f"Processing {symbol}")
                try:
                    symbol_data = data[data['symbol'] == symbol]
                    symbol_features = features.xs(symbol, level='symbol')
                    results[symbol] = self._run_symbol_backtest(symbol, symbol_data, symbol_features)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Save and plot results
            self._save_and_plot_results(results, market, timeframe)
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")

    def _prepare_data(self, market: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting"""
        try:
            # Load data
            data = self.data_loader.get_merged_data(market, timeframe)
            if data.empty:
                logger.error("No data loaded")
                return None
            
            # Generate target variable
            data['target'] = (data.groupby('symbol')['close']
                            .shift(-1)
                            .gt(data['close'])
                            .astype(int))
            
            # Drop last row for each symbol as it has NaN target
            data = data.groupby('symbol').apply(lambda x: x[:-1]).reset_index(drop=True)
            
            logger.info(f"Prepared data shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def _run_symbol_backtest(self, symbol: str, data: pd.DataFrame, 
                           features: pd.DataFrame) -> Optional[Dict]:
        """Run backtest for a single symbol"""
        try:
            # Validate data
            if len(data) != len(features):
                logger.error(f"Data length mismatch for {symbol}")
                return None
                
            train_size = int(len(data) * self.config['training']['train_size'])
            if train_size < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Split data
            train_features = features.iloc[:train_size]
            train_target = data['target'].iloc[:train_size]
            test_features = features.iloc[train_size:]
            test_data = data.iloc[train_size:]
            
            # Check for NaN values
            if train_features.isna().any().any() or test_features.isna().any().any():
                logger.warning(f"NaN values found in features for {symbol}")
                train_features = train_features.fillna(0)
                test_features = test_features.fillna(0)
            
            # Train model
            model = self._train_model(train_features, train_target)
            
            # Generate predictions and calculate positions
            predictions = model.predict_proba(test_features)[:, 1]
            position_sizes = self._calculate_position_sizes(
                predictions, 
                test_data['close'],
                test_features['volatility_20']
            )
            
            # Calculate returns
            returns = self._calculate_returns(test_data['close'], position_sizes)
            
            return {
                'returns': returns,
                'predictions': predictions,
                'position_sizes': position_sizes,
                'close_prices': test_data['close'],
                'features': test_features,
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Error in symbol backtest for {symbol}: {str(e)}")
            return None

    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> lgb.LGBMClassifier:
        """Train LightGBM model with early stopping"""
        try:
            # Split training data for validation
            train_size = int(len(features) * 0.8)
            train_features = features.iloc[:train_size]
            train_target = target.iloc[:train_size]
            val_features = features.iloc[train_size:]
            val_target = target.iloc[train_size:]
            
            # Initialize and train model
            model = lgb.LGBMClassifier(**self.config['model']['params'])
            model.fit(
                train_features, train_target,
                eval_set=[(val_features, val_target)],
                early_stopping_rounds=50,
                verbose=False
            )
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _calculate_position_sizes(self, predictions: np.ndarray, 
                                prices: pd.Series,
                                volatility: pd.Series) -> pd.Series:
        """Calculate position sizes based on predictions and volatility"""
        try:
            # Create base positions
            sizes = pd.Series(0, index=prices.index)
            sizes[predictions > 0.7] = 1
            sizes[predictions < 0.3] = -1
            
            # Scale by volatility
            vol_scale = volatility.clip(lower=0).rolling(20).mean()
            position_scale = (1 - vol_scale/vol_scale.max()).fillna(0.5)
            
            return sizes * position_scale
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {str(e)}")
            return pd.Series(0, index=prices.index)

    def _calculate_returns(self, prices: pd.Series, positions: pd.Series) -> pd.Series:
        """Calculate strategy returns including transaction costs"""
        try:
            price_changes = prices.pct_change()
            position_changes = positions.diff().abs() * self.config['portfolio']['transaction_cost']
            return (price_changes * positions) - position_changes
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.Series(0, index=prices.index)

    def _save_and_plot_results(self, results: Dict, market: str, timeframe: str) -> None:
        """Save and plot backtest results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results
            filename = f'{self.results_dir}/backtest_{market}_{timeframe}_{timestamp}.pkl'
            pd.to_pickle(results, filename)
            logger.info(f"Results saved to {filename}")
            
            # Create plots
            self._plot_results(results, market, timeframe, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving/plotting results: {str(e)}")

    def _plot_results(self, results: Dict, market: str, timeframe: str, timestamp: str) -> None:
        """Create visualization of backtest results"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            
            # Plot cumulative returns
            for symbol, result in results.items():
                if result is not None:
                    cumret = (1 + result['returns']).cumprod()
                    axes[0].plot(cumret.index, cumret.values, label=symbol)
            
            axes[0].set_title('Cumulative Returns')
            axes[0].legend(bbox_to_anchor=(1.05, 1))
            axes[0].grid(True)
            
            # Plot position sizes
            for symbol, result in results.items():
                if result is not None:
                    axes[1].plot(result['position_sizes'].index, 
                               result['position_sizes'].values,
                               label=symbol, alpha=0.5)
            
            axes[1].set_title('Position Sizes')
            axes[1].legend(bbox_to_anchor=(1.05, 1))
            axes[1].grid(True)
            
            # Plot prediction distributions
            for symbol, result in results.items():
                if result is not None:
                    axes[2].hist(result['predictions'], bins=50, 
                               alpha=0.3, label=symbol)
            
            axes[2].set_title('Prediction Distributions')
            axes[2].legend(bbox_to_anchor=(1.05, 1))
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/backtest_{market}_{timeframe}_{timestamp}.png',
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Plots saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")

def main():
    """Main execution function"""
    try:
        logger.info("Starting ML backtesting")
        backtest = Backtest(ML_CONFIG)
        
        # Run backtests
        for market in ['stocks', 'crypto']:
            backtest.run_backtest(market, 'daily')
            
        logger.info("Backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
