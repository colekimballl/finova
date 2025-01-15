# progressive_training.py

import logging
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, List
import json
from pathlib import Path

# Local imports
from risk_manager import RiskManager, RiskConfig
from ml_ensemble import EnhancedMLModel, MarketRegimeDetector
from data_scripts.data_loader import MarketDataLoader
from feature_generator import FeatureGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressiveTraining:
    def __init__(self, config_path: str = 'config/training_config.json'):
        """Initialize progressive training framework"""
        self.config = self._load_config(config_path)
        self.results_dir = Path('results')
        self.models_dir = Path('models')
        self.data_loader = MarketDataLoader()
        
        # Create necessary directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.risk_manager = RiskManager(RiskConfig(**self.config['risk']))
        self.feature_generator = FeatureGenerator()
        self.regime_detector = MarketRegimeDetector()
        
        # Training history
        self.training_history = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def run_progressive_training(self, 
                               market: str = 'stocks',
                               timeframe: str = 'daily',
                               n_iterations: int = 5):
        """Run progressive training with feedback loops"""
        
        # Initial data load and preparation
        data = self._load_and_prepare_data(market, timeframe)
        if data is None:
            return
            
        for iteration in tqdm(range(n_iterations), desc='Training Iterations'):
            logger.info(f"Starting iteration {iteration + 1}/{n_iterations}")
            
            # 1. Generate features with current knowledge
            features = self.feature_generator.generate_features(data)
            
            # 2. Detect market regimes
            regimes = self.regime_detector.detect_regime(data)
            
            # 3. Create and train model for this iteration
            model = EnhancedMLModel(self.config['model'])
            
            # 4. Perform cross-validation training
            cv_metrics = self._cross_validation_train(
                model, features, data['target'].values, regimes
            )
            
            # 5. Run backtest with current model
            backtest_results = self._run_backtest(
                model, data, features, regimes
            )
            
            # 6. Update feature importance and strategy parameters
            self._update_strategy(model, backtest_results)
            
            # 7. Save progress
            self._save_iteration_results(
                iteration, model, cv_metrics, backtest_results
            )
            
            # 8. Log progress
            self._log_progress(iteration, cv_metrics, backtest_results)
            
    def _cross_validation_train(self, 
                              model: EnhancedMLModel,
                              features: pd.DataFrame,
                              targets: np.ndarray,
                              regimes: np.ndarray) -> Dict:
        """Perform time series cross-validation training"""
        from sklearn.model_selection import TimeSeriesSplit
        
        cv_results = []
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            # Train model
            train_metrics = model.train(
                features.iloc[train_idx],
                targets[train_idx],
                regimes[train_idx]
            )
            
            # Validate
            val_metrics = self._validate_model(
                model,
                features.iloc[val_idx],
                targets[val_idx],
                regimes[val_idx]
            )
            
            cv_results.append({
                'train': train_metrics,
                'validation': val_metrics
            })
            
        return cv_results
        
    def _run_backtest(self,
                     model: EnhancedMLModel,
                     data: pd.DataFrame,
                     features: pd.DataFrame,
                     regimes: np.ndarray) -> Dict:
        """Run backtest with current model"""
        results = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_features = features.loc[symbol_data.index]
            symbol_regimes = regimes[data['symbol'] == symbol]
            
            # Generate predictions
            predictions = []
            confidences = []
            
            for i in range(len(symbol_features)):
                pred, conf = model.predict(
                    symbol_features.iloc[i:i+1],
                    symbol_regimes[i]
                )
                predictions.append(pred)
                confidences.append(conf)
                
            # Calculate positions with risk management
            positions = []
            for j in range(len(predictions)):
                position = self.risk_manager.calculate_position_size(
                    symbol,
                    symbol_data.iloc[:j+1],
                    predictions[j],
                    confidences[j]
                )
                position, stopped = self.risk_manager.check_stops(
                    symbol,
                    symbol_data['close'].iloc[j],
                    position
                )
                positions.append(position)
                
            # Calculate returns
            symbol_returns = symbol_data['close'].pct_change()
            strategy_returns = pd.Series(positions).shift() * symbol_returns
            
            results[symbol] = {
                'predictions': predictions,
                'confidences': confidences,
                'positions': positions,
                'returns': strategy_returns,
                'equity_curve': (1 + strategy_returns).cumprod()
            }
            
        return results
        
    def _update_strategy(self, 
                        model: EnhancedMLModel,
                        backtest_results: Dict):
        """Update strategy based on backtest results"""
        # Get top features
        top_features = model.get_top_features()
        
        # Update feature generator weights
        self.feature_generator.update_weights(top_features)
        
        # Update risk parameters based on performance
        self._adjust_risk_parameters(backtest_results)
        
    def _adjust_risk_parameters(self, backtest_results: Dict):
        """Adjust risk parameters based on performance"""
        # Calculate overall Sharpe ratio
        all_returns = pd.concat([
            res['returns'] for res in backtest_results.values()
        ])
        sharpe = np.sqrt(252) * all_returns.mean() / all_returns.std()
        
        # Adjust risk parameters
        if sharpe < 0.5:  # If performance is poor
            self.risk_manager.config.max_position_size *= 0.9
            self.risk_manager.config.stop_loss_pct *= 0.9
        elif sharpe > 2.0:  # If performance is very good
            self.risk_manager.config.max_position_size = min(
                1.0,
                self.risk_manager.config.max_position_size * 1.1
            )
            
    def _save_iteration_results(self,
                              iteration: int,
                              model: EnhancedMLModel,
                              cv_metrics: Dict,
                              backtest_results: Dict):
        """Save results for current iteration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = self.models_dir / f'model_iteration_{iteration}_{timestamp}'
        model.save(model_path)
        
        # Save metrics
        metrics_path = self.results_dir / f'metrics_iteration_{iteration}_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'cv_metrics': cv_metrics,
                'backtest_metrics': self._calculate_backtest_metrics(backtest_results)
            }, f, indent=4)
            
    def _log_progress(self,
                     iteration: int,
                     cv_metrics: Dict,
                     backtest_results: Dict):
        """Log training progress"""
        logger.info(f"\nIteration {iteration + 1} Results:")
        logger.info("-" * 50)
        
        # Log cross-validation metrics
        mean_cv_f1 = np.mean([
            m['validation']['f1'] for m in cv_metrics
        ])
        logger.info(f"Mean CV F1 Score: {mean_cv_f1:.4f}")
        
        # Log backtest metrics
        backtest_metrics = self._calculate_backtest_metrics(backtest_results)
        logger.info(f"Backtest Sharpe Ratio: {backtest_metrics['sharpe']:.4f}")
        logger.info(f"Backtest Max Drawdown: {backtest_metrics['max_drawdown']:.4f}")
        logger.info(f"Win Rate: {backtest_metrics['win_rate']:.4f}")
        
    def _calculate_backtest_metrics(self, backtest_results: Dict) -> Dict:
        """Calculate overall backtest metrics"""
        all_returns = pd.concat([
            res['returns'] for res in backtest_results.values()
        ])
        
        return {
            'sharpe': np.sqrt(252) * all_returns.mean() / all_returns.std(),
            'max_drawdown': self._calculate_max_drawdown(all_returns),
            'win_rate': (all_returns > 0).mean(),
            'profit_factor': abs(
                all_returns[all_returns > 0].sum() /
                all_returns[all_returns < 0].sum()
            )
        }
        
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        return abs(drawdowns.min())

def main():
    """Main execution"""
    trainer = ProgressiveTraining()
    
    # Run progressive training for both markets
    for market in ['stocks', 'crypto']:
        logger.info(f"\nStarting progressive training for {market}")
        trainer.run_progressive_training(market=market, n_iterations=5)

if __name__ == "__main__":
    main()
