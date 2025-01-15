# ml_testing.py

import logging
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from typing import Dict, Optional
from data_scripts.data_loader import MarketDataLoader
from feature_generator import FeatureGenerator
from ml_config import ML_CONFIG

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLBacktest:
    def __init__(self):
        """Initialize ML backtesting framework"""
        self.config = ML_CONFIG
        self.data_loader = MarketDataLoader()
        self.feature_generator = FeatureGenerator(self.config)
        self.results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.initial_capital = self.config['portfolio']['initial_capital']
        
    def run(self, market: str = 'crypto', timeframe: str = 'daily'):
        """Run complete backtest process"""
        try:
            # 1. Load and prepare data
            logger.info(f"Starting backtest for {market} {timeframe}")
            data = self._load_and_prepare_data(market, timeframe)
            if data is None:
                return
                
            # 2. Generate features
            features = self._generate_features(data)
            if features is None:
                return
                
            # 3. Run individual symbol backtests
            results = {}
            for symbol in data['symbol'].unique():
                logger.info(f"Processing {symbol}")
                result = self._backtest_symbol(symbol, data, features)
                if result is not None:
                    results[symbol] = result
                    
            # 4. Save and visualize results
            self._save_results(results, market, timeframe)
            self._plot_results(results, market, timeframe)
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}", exc_info=True)

    def _load_and_prepare_data(self, market: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load and prepare data for backtesting"""
        try:
            # Load data
            data = self.data_loader.get_merged_data(market, timeframe)
            if data.empty:
                logger.error("No data loaded")
                return None
                
            # Add target variable (next day return direction)
            data['target'] = (data.groupby('symbol')['close']
                            .shift(-1)
                            .gt(data['close'])
                            .astype(int))
            
            # Remove last row of each symbol (has NaN target)
            data = data.groupby('symbol').apply(lambda x: x[:-1]).reset_index(drop=True)
            
            logger.info(f"Prepared data shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def _generate_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features for ML model"""
        try:
            features = self.feature_generator.generate_features(data)
            if features is None:
                logger.error("Feature generation failed")
                return None
                
            # Handle NaN values
            features = features.fillna(method='ffill').fillna(0)
            return features
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            return None

    def _backtest_symbol(self, symbol: str, data: pd.DataFrame, features: pd.DataFrame) -> Optional[Dict]:
        """Run backtest for a single symbol"""
        try:
            # Get symbol specific data
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_features = features.xs(symbol, level='symbol')
            
            # Split data
            train_size = int(len(symbol_data) * self.config['training']['train_size'])
            
            # Train model
            model = self._train_model(
                symbol_features.iloc[:train_size],
                symbol_data['target'].iloc[:train_size]
            )
            
            # Generate predictions
            predictions = model.predict_proba(symbol_features.iloc[train_size:])[:, 1]
            
            # Calculate positions and returns
            positions = self._calculate_positions(predictions, symbol_data.iloc[train_size:])
            returns = self._calculate_returns(positions, symbol_data.iloc[train_size:])
            
            return {
                'predictions': predictions,
                'positions': positions,
                'returns': returns,
                'model': model,
                'prices': symbol_data['close'].iloc[train_size:],
                'dates': symbol_data.index[train_size:]
            }
            
        except Exception as e:
            logger.error(f"Error in symbol backtest for {symbol}: {str(e)}")
            return None

    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        model = lgb.LGBMClassifier(**self.config['model']['params'])
        model.fit(features, target)
        return model

    def _calculate_positions(self, predictions: np.ndarray, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes based on predictions"""
        # Convert predictions to positions (-1 to 1)
        positions = pd.Series(0, index=data.index)
        positions[predictions > 0.7] = 1
        positions[predictions < 0.3] = -1
        
        # Apply position sizing based on volatility
        vol = data['close'].pct_change().rolling(20).std()
        vol_scale = (1 - vol/vol.max()).fillna(0.5)
        
        return positions * vol_scale

    def _calculate_returns(self, positions: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns"""
        price_returns = data['close'].pct_change()
        position_changes = positions.diff().abs() * self.config['portfolio']['transaction_cost']
        return (positions.shift() * price_returns) - position_changes

    def _save_results(self, results: Dict, market: str, timeframe: str):
        """Save backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.results_dir}/backtest_{market}_{timeframe}_{timestamp}.pkl'
        pd.to_pickle(results, filename)
        
        # Save performance metrics
        metrics = self._calculate_metrics(results)
        metrics_file = f'{self.results_dir}/metrics_{market}_{timeframe}_{timestamp}.csv'
        pd.DataFrame(metrics).to_csv(metrics_file)
        
        logger.info(f"Results saved to {self.results_dir}")

    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        for symbol, result in results.items():
            if result is not None:
                rets = result['returns']
                metrics[symbol] = {
                    'Total Return': (1 + rets).prod() - 1,
                    'Annual Return': (1 + rets).prod() ** (252/len(rets)) - 1,
                    'Sharpe Ratio': np.sqrt(252) * rets.mean() / rets.std(),
                    'Max Drawdown': (1 - (1 + rets).cumprod() / 
                                   (1 + rets).cumprod().cummax()).max(),
                    'Win Rate': (rets > 0).mean(),
                    'Profit Factor': abs(rets[rets > 0].sum() / rets[rets < 0].sum())
                }
        return metrics

    def _plot_results(self, results: Dict, market: str, timeframe: str):
        """Create visualization of backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up the plot
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        for symbol, result in results.items():
            cumret = (1 + result['returns']).cumprod()
            ax1.plot(cumret.index, cumret.values, label=symbol)
        ax1.set_title('Cumulative Returns', fontsize=12)
        ax1.legend()
        
        # 2. Position Sizes
        ax2 = fig.add_subplot(gs[1, 0])
        for symbol, result in results.items():
            ax2.plot(result['positions'].index, 
                    result['positions'].values,
                    label=symbol, alpha=0.5)
        ax2.set_title('Position Sizes', fontsize=12)
        
        # 3. Predictions Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for symbol, result in results.items():
            ax3.hist(result['predictions'], bins=50,
                    alpha=0.3, label=symbol)
        ax3.set_title('Prediction Distribution', fontsize=12)
        
        # 4. Performance Metrics Heatmap
        metrics = pd.DataFrame(self._calculate_metrics(results)).T
        ax4 = fig.add_subplot(gs[2, :])
        sns.heatmap(metrics, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Performance Metrics', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/backtest_{market}_{timeframe}_{timestamp}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Main execution"""
    backtest = MLBacktest()
    
    # Run for both markets
    for market in ['stocks', 'crypto']:
        backtest.run(market, 'daily')
        
    logger.info("Backtesting completed")

if __name__ == "__main__":
    main()
