# progressive_training.py

import logging
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Local imports (Ensure these modules are correctly implemented)
from risk_manager import RiskManager, RiskConfig
from ml_ensemble import EnhancedMLModel
from config.training_config import CONFIG
import sys
sys.path.append("../../")
from data_scripts.data_loader import MarketDataLoader

# Configure logging with immediate flushing
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Configure logging
logs_dir = Path('../../logs/v2')
logs_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = FlushStreamHandler(sys.stdout)
file_handler = logging.FileHandler(logs_dir / 'v2_training.log', mode='a', encoding='utf-8')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

class ProgressiveTraining:
    def __init__(self, quick_test: bool = False):
        """
        Initialize the progressive training framework.
        
        Args:
            quick_test (bool): If True, runs in quick test mode with reduced data.
        """
        self.config = CONFIG
        self.quick_test = quick_test
        self.results_dir = Path('../../results/v2')
        self.models_dir = Path('../../models/v2')
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.risk_manager = RiskManager(RiskConfig(**self.config.get('risk', {})))
        self.data_loader = MarketDataLoader()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.history = {
            'metrics': [],
            'feature_importances': [],
            'model_weights': []
        }

    def run_progressive_training(self, 
                                 market: str = 'stocks',
                                 timeframe: str = 'daily',
                                 n_iterations: int = 5):
        """
        Run progressive training with feedback loops.
        
        Args:
            market (str): Market to train on ('stocks' or 'crypto').
            timeframe (str): Timeframe for analysis ('daily' or 'hourly').
            n_iterations (int): Number of training iterations.
        """
        start_time = datetime.now()
        logger.info("\n" + "="*50)
        logger.info(f"Starting progressive training for {market} {timeframe}")
        logger.info(f"Quick test mode: {self.quick_test}")
        logger.info("="*50 + "\n")
        
        try:
            # Load and prepare data
            data = self._load_and_prepare_data(market, timeframe)
            if data is None:
                logger.error("Data loading failed. Exiting training.")
                return
                
            # Main training loop
            for iteration in range(n_iterations):
                iteration_start = datetime.now()
                logger.info(f"\nIteration {iteration + 1}/{n_iterations}")
                logger.info("-" * 50)
                
                # Run single iteration
                try:
                    results = self._run_single_iteration(data, iteration)
                    
                    # Save and visualize results
                    self._save_iteration_results(results, iteration, market, timeframe)
                    
                    # Log iteration summary
                    duration = datetime.now() - iteration_start
                    logger.info(f"\nIteration {iteration + 1} completed in {duration}")
                    self._log_iteration_summary(results, iteration)
                    
                except Exception as e:
                    logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                    continue
                
            # Final summary
            total_duration = datetime.now() - start_time
            logger.info(f"\nTraining completed in {total_duration}")
            self._create_training_summary(market, timeframe)
            self._generate_comprehensive_report(market, timeframe)
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def _load_and_prepare_data(self, market: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load and prepare data with progress tracking.
        
        Args:
            market (str): Market to load data for.
            timeframe (str): Timeframe of the data.
        
        Returns:
            Optional[pd.DataFrame]: Prepared data or None if loading fails.
        """
        logger.info("Loading market data...")
        data = self.data_loader.get_merged_data(market, timeframe)
        
        if data.empty:
            logger.error("No data loaded.")
            return None
            
        logger.info(f"Loaded {len(data)} rows for {len(data['symbol'].unique())} symbols.")
        
        if self.quick_test:
            # Use only top 5 symbols by data length for quick testing
            symbol_counts = data['symbol'].value_counts().head(5)
            logger.info(f"Quick test mode - using symbols: {', '.join(symbol_counts.index)}")
            data = data[data['symbol'].isin(symbol_counts.index)]
        
        # Add target variable
        data['target'] = (data.groupby('symbol')['close']
                         .shift(-1)
                         .gt(data['close'])
                         .astype(int))
        
        # Remove last row for each symbol by dropping rows with NaN target
        data = data.dropna(subset=['target']).reset_index(drop=True)
        
        logger.info(f"Final data shape: {data.shape}")
        return data

    def _run_single_iteration(self, data: pd.DataFrame, iteration: int) -> Dict:
        """
        Run a single training iteration.
        
        Args:
            data (pd.DataFrame): Prepared data.
            iteration (int): Current iteration number.
        
        Returns:
            Dict: Dictionary containing trained model, cross-validation metrics, backtest results, and features.
        """
        # 1. Detect market regimes
        logger.info("Detecting market regimes...")
        regimes = self._detect_market_regimes(data)
        
        # 2. Generate features
        logger.info("Generating features...")
        features = self._generate_features(data)
        
        # 3. Train model
        logger.info("Training models...")
        model = EnhancedMLModel(self.config)
        cv_metrics = model.train(features, data['target'], regimes)
        
        # 4. Run backtest
        logger.info("Running backtest...")
        backtest_results = self._run_backtest(model, data, features, regimes)
        
        return {
            'model': model,
            'cv_metrics': cv_metrics,
            'backtest_results': backtest_results,
            'features': features
        }

    def _detect_market_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect market regimes based on the data.
        
        Args:
            data (pd.DataFrame): Prepared data.
        
        Returns:
            np.ndarray: Array indicating market regimes.
        """
        # Placeholder for market regime detection logic
        # For demonstration, we'll use a simple moving average crossover strategy
        logger.info("Calculating regime features...")
        short_ma = data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        long_ma = data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=200, min_periods=1).mean())
        regimes = (short_ma > long_ma).astype(int).values
        logger.info(f"Detected regimes shape: {regimes.shape}")
        return regimes

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features with progress tracking.
        
        Args:
            data (pd.DataFrame): Prepared data.
        
        Returns:
            pd.DataFrame: Generated feature matrix.
        """
        features_list = []
        
        # Price-based features
        for window in tqdm([5, 10, 20, 50], desc="Generating price features"):
            pct_change = data.groupby('symbol')['close'].pct_change(window).rename(f'return_{window}d')
            rolling_mean = data.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True).rename(f'ma_{window}d')
            rolling_std = data.groupby('symbol')['close'].rolling(window).std().reset_index(0, drop=True).rename(f'vol_{window}d')
            features_list.extend([pct_change, rolling_mean, rolling_std])
        
        # Volume features
        volume_change = data.groupby('symbol')['volume'].pct_change().rename('volume_change')
        volume_ma_ratio = (data['volume'] / data.groupby('symbol')['volume']
                           .rolling(20).mean().reset_index(0, drop=True)).rename('volume_ma_ratio')
        features_list.extend([volume_change, volume_ma_ratio])
        
        # Technical indicators
        logger.info("Calculating technical indicators...")
        # RSI
        delta = data.groupby('symbol')['close'].diff()
        gain = (delta.where(delta > 0, 0)).reset_index(0, drop=True)
        loss = (-delta.where(delta < 0, 0)).reset_index(0, drop=True)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.replace([np.inf, -np.inf], 0).fillna(0)  # Handle potential infs and NaNs
        features_list.append(pd.Series(rsi, name='rsi'))
        
        # Combine features
        features = pd.concat(features_list, axis=1)
        
        # Replace infs and NaNs
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Clip values to prevent extremely large numbers
        features = features.clip(-10, 10)
        
        # Ensure all column names are strings
        features.columns = features.columns.astype(str)
        
        logger.info(f"Generated {features.shape[1]} features.")
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        features = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
        
        return features

    def _run_backtest(self, 
                      model: EnhancedMLModel,
                      data: pd.DataFrame,
                      features: pd.DataFrame,
                      regimes: np.ndarray) -> Dict:
        """
        Run backtest with the trained model.
        
        Args:
            model (EnhancedMLModel): Trained ensemble model.
            data (pd.DataFrame): Prepared data.
            features (pd.DataFrame): Generated feature matrix.
            regimes (np.ndarray): Array indicating market regimes.
        
        Returns:
            Dict: Backtest results for each symbol.
        """
        results = {}
        symbols = data['symbol'].unique()
        
        logger.info("Running backtests in parallel.")
        
        # Prepare data for parallel processing
        symbol_data_list = [data[data['symbol'] == symbol].copy() for symbol in symbols]
        symbol_features_list = [features[data['symbol'] == symbol].copy() for symbol in symbols]
        symbol_regimes_list = [regimes[data['symbol'] == symbol] for symbol in symbols]
        
        # Define backtest function for parallel processing
        def backtest_symbol(symbol, symbol_data, symbol_features, symbol_regimes, model, risk_manager) -> Tuple[str, Dict]:
            """
            Backtest a single symbol.
            
            Args:
                symbol (str): Symbol name.
                symbol_data (pd.DataFrame): Data for the symbol.
                symbol_features (pd.DataFrame): Features for the symbol.
                symbol_regimes (np.ndarray): Regimes for the symbol.
                model (EnhancedMLModel): Trained model.
                risk_manager (RiskManager): Risk manager instance.
            
            Returns:
                Tuple[str, Dict]: Symbol name and its backtest results.
            """
            try:
                logger.info(f"Backtesting symbol: {symbol}")
                
                # Make bulk predictions
                ensemble_pred, confidence = model.predict_bulk(symbol_features, symbol_regimes)
                
                # Calculate positions
                positions = risk_manager.calculate_position_size_bulk(
                    symbol,
                    symbol_data,
                    ensemble_pred,
                    confidence
                )
                
                # Apply stops
                positions, stopped = risk_manager.apply_stops_bulk(
                    symbol,
                    symbol_data['close'],
                    positions
                )
                
                # Calculate returns
                returns = positions.shift() * symbol_data['close'].pct_change()
                
                # Calculate equity curve
                equity_curve = (1 + returns).cumprod()
                
                # Compile results
                symbol_result = {
                    'predictions': ensemble_pred.tolist(),
                    'confidences': confidence.tolist(),
                    'positions': positions.tolist(),
                    'returns': returns.tolist(),
                    'equity_curve': equity_curve.tolist()
                }
                
                logger.info(f"Completed backtest for symbol: {symbol}")
                return (symbol, symbol_result)
            
            except Exception as e:
                logger.error(f"Error backtesting symbol {symbol}: {e}")
                return (symbol, {})
        
        # Parallel processing of backtests
        processed_results = Parallel(n_jobs=-1)(
            delayed(backtest_symbol)(
                symbol, 
                symbol_data, 
                symbol_features, 
                symbol_regimes, 
                model, 
                self.risk_manager
            )
            for symbol, symbol_data, symbol_features, symbol_regimes in zip(symbols, symbol_data_list, symbol_features_list, symbol_regimes_list)
        )
        
        # Aggregate results
        for symbol, res in processed_results:
            if res:
                results[symbol] = res
        
        return results

    def _calculate_metrics(self, results: Dict) -> Dict:
        """
        Calculate performance metrics for backtest results.
        
        Args:
            results (Dict): Backtest results for each symbol.
        
        Returns:
            Dict: Performance metrics for each symbol.
        """
        metrics = {}
        for symbol, result in results.items():
            returns = pd.Series(result['returns']).fillna(0)
            if returns.std() == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
            max_drawdown = (1 - (1 + returns).cumprod() / (1 + returns).cumprod().cummax()).max()
            win_rate = (returns > 0).mean()
            profit_factor = abs(returns[returns > 0].sum() / (returns[returns < 0].sum() or 1e-10))
            
            metrics[symbol] = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Profit Factor': profit_factor
            }
        return metrics

    def _log_iteration_summary(self, results: Dict, iteration: int):
        """
        Log summary of iteration results.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
        """
        metrics = self._calculate_metrics(results['backtest_results'])
        if not metrics:
            logger.warning("No metrics to summarize for this iteration.")
            return
        avg_metrics = pd.DataFrame(metrics).mean()
        
        logger.info("\nIteration Summary:")
        logger.info(f"Average Sharpe Ratio: {avg_metrics['Sharpe Ratio']:.2f}")
        logger.info(f"Average Annual Return: {avg_metrics['Annual Return']:.2%}")
        logger.info(f"Average Max Drawdown: {avg_metrics['Max Drawdown']:.2%}")
        logger.info(f"Average Win Rate: {avg_metrics['Win Rate']:.2%}")
        logger.info(f"Profit Factor: {avg_metrics['Profit Factor']:.2f}")
        logger.info(f"Model Weights: {results['model'].model_weights}")
        
        # Log cross-validation metrics if available
        if results['cv_metrics']:
            try:
                avg_f1 = {name: np.mean(scores) for name, scores in results['cv_metrics'].items()}
                for name, f1 in avg_f1.items():
                    logger.info(f"Average CV F1 Score for {name.upper()}: {f1:.4f}")
            except Exception as e:
                logger.error(f"Error calculating average CV F1 Score: {str(e)}")

    def _save_iteration_results(self, 
                                results: Dict,
                                iteration: int,
                                market: str,
                                timeframe: str):
        """
        Save iteration results and create visualizations.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Define model path with proper extension
        model_base_path = self.models_dir / f'model_{market}_{timeframe}_iter_{iteration}_{timestamp}'
        
        # Save model
        try:
            results['model'].save(str(model_base_path))
            logger.info(f"Saved model to {model_base_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        # Save scaler
        scaler_path = model_base_path.with_suffix('.scaler.pkl')
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
        
        # Create and save visualizations
        self._create_iteration_plots(results, iteration, market, timeframe, timestamp)
        
        # Update history
        self.history['metrics'].append(self._calculate_metrics(results['backtest_results']))
        self.history['feature_importances'].append(results['model'].feature_importances)
        self.history['model_weights'].append(results['model'].model_weights)

    def _create_iteration_plots(self,
                                 results: Dict,
                                 iteration: int,
                                 market: str,
                                 timeframe: str,
                                 timestamp: str):
        """
        Create detailed plots for the iteration.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
            timestamp (str): Current timestamp.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for symbol, result in results['backtest_results'].items():
            equity_curve = pd.Series(result['equity_curve'])
            ax1.plot(equity_curve.values, label=symbol)
        ax1.set_title(f'Equity Curves - Iteration {iteration + 1}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # 2. Position sizes over time
        ax2 = fig.add_subplot(gs[1, 0])
        for symbol, result in results['backtest_results'].items():
            positions = pd.Series(result['positions'])
            ax2.plot(positions, label=symbol, alpha=0.5)
        ax2.set_title('Position Sizes')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        # 3. Prediction distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for symbol, result in results['backtest_results'].items():
            predictions = pd.Series(result['predictions'])
            sns.histplot(predictions, bins=50, kde=True, label=symbol, ax=ax3, alpha=0.5)
        ax3.set_title('Prediction Distribution')
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # 4. Performance metrics heatmap
        ax4 = fig.add_subplot(gs[2, :])
        metrics = pd.DataFrame(self._calculate_metrics(results['backtest_results'])).T
        if not metrics.empty:
            sns.heatmap(metrics, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
            ax4.set_title('Performance Metrics')
        else:
            logger.warning("No performance metrics available to plot.")
            ax4.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        plot_filename = self.results_dir / f'iteration_{iteration}_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved iteration plot to {plot_filename}")

    def _create_training_summary(self, market: str, timeframe: str):
        """
        Create and save training summary visualizations.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.history['metrics']:
            logger.warning("No training metrics available to create summary.")
            return
        
        # Create summary plots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Performance metrics over iterations
        ax1 = fig.add_subplot(gs[0, :])
        metrics_df = pd.DataFrame([
            {k: np.mean([m[k] for m in metrics.values()])
             for k in next(iter(metrics.values())).keys()}
            for metrics in self.history['metrics']
        ])
        if not metrics_df.empty:
            metrics_df.plot(ax=ax1, marker='o')
            ax1.set_title('Performance Metrics Across Iterations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True)
        else:
            logger.warning("No performance metrics available to plot across iterations.")
            ax1.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Feature importance evolution
        ax2 = fig.add_subplot(gs[1, :])
        if self.history['feature_importances']:
            importance_df = pd.DataFrame(self.history['feature_importances'])
            sns.heatmap(importance_df.T, ax=ax2, cmap='YlOrRd', annot=False)
            ax2.set_title('Feature Importance Evolution')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Feature')
        else:
            logger.warning("No feature importances available to plot.")
            ax2.text(0.5, 0.5, 'No Feature Importances Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. Model weights evolution
        ax3 = fig.add_subplot(gs[2, :])
        if self.history['model_weights']:
            weights_df = pd.DataFrame(self.history['model_weights'])
            weights_df.plot(kind='bar', ax=ax3)
            ax3.set_title('Model Weights Evolution')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Weight')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
        else:
            logger.warning("No model weights available to plot.")
            ax3.text(0.5, 0.5, 'No Model Weights Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        plt.tight_layout()
        summary_plot_filename = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            summary_plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved training summary plot to {summary_plot_filename}")
        
        # Save summary metrics to CSV
        if not self.history['metrics']:
            logger.warning("No training metrics available to save as CSV.")
        else:
            summary_csv_path = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.csv'
            pd.DataFrame(self.history['metrics']).to_csv(summary_csv_path, index=False)
            logger.info(f"Saved training summary metrics to {summary_csv_path}")

    def _generate_comprehensive_report(self, market: str, timeframe: str):
        """
        Generate a comprehensive report in .txt and .csv formats.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_txt_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.txt'
        report_csv_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.csv'
        
        try:
            with open(report_txt_path, 'w') as f:
                f.write(f"Training Report - {market.capitalize()} {timeframe.capitalize()}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for idx, metrics in enumerate(self.history['metrics'], 1):
                    f.write(f"--- Iteration {idx} ---\n")
                    for symbol, symbol_metrics in metrics.items():
                        f.write(f"Symbol: {symbol}\n")
                        for metric_name, value in symbol_metrics.items():
                            if isinstance(value, float):
                                if 'Return' in metric_name or 'Drawdown' in metric_name:
                                    f.write(f"  {metric_name}: {value:.2%}\n")
                                else:
                                    f.write(f"  {metric_name}: {value:.4f}\n")
                            else:
                                f.write(f"  {metric_name}: {value}\n")
                        f.write("\n")
                    f.write(f"Model Weights: {self.history['model_weights'][idx-1]}\n\n")
            logger.info(f"Saved comprehensive report to {report_txt_path}")
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
        
        try:
            # Convert metrics to a single DataFrame
            all_metrics = []
            for idx, metrics in enumerate(self.history['metrics'], 1):
                for symbol, symbol_metrics in metrics.items():
                    row = {'Iteration': idx, 'Symbol': symbol}
                    row.update(symbol_metrics)
                    row.update({
                        'Model Weight_rf': self.history['model_weights'][idx-1].get('rf', np.nan),
                        'Model Weight_gb': self.history['model_weights'][idx-1].get('gb', np.nan),
                        'Model Weight_nn': self.history['model_weights'][idx-1].get('nn', np.nan)
                    })
                    all_metrics.append(row)
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(report_csv_path, index=False)
            logger.info(f"Saved comprehensive report CSV to {report_csv_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

    def _log_iteration_summary(self, results: Dict, iteration: int):
        """
        Log summary of iteration results.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
        """
        metrics = self._calculate_metrics(results['backtest_results'])
        if not metrics:
            logger.warning("No metrics to summarize for this iteration.")
            return
        avg_metrics = pd.DataFrame(metrics).mean()
        
        logger.info("\nIteration Summary:")
        logger.info(f"Average Sharpe Ratio: {avg_metrics['Sharpe Ratio']:.2f}")
        logger.info(f"Average Annual Return: {avg_metrics['Annual Return']:.2%}")
        logger.info(f"Average Max Drawdown: {avg_metrics['Max Drawdown']:.2%}")
        logger.info(f"Average Win Rate: {avg_metrics['Win Rate']:.2%}")
        logger.info(f"Profit Factor: {avg_metrics['Profit Factor']:.2f}")
        logger.info(f"Model Weights: {results['model'].model_weights}")
        
        # Log cross-validation metrics if available
        if results['cv_metrics']:
            try:
                avg_f1 = {name: np.mean(scores) for name, scores in results['cv_metrics'].items()}
                for name, f1 in avg_f1.items():
                    logger.info(f"Average CV F1 Score for {name.upper()}: {f1:.4f}")
            except Exception as e:
                logger.error(f"Error calculating average CV F1 Score: {str(e)}")

    def _save_iteration_results(self, 
                                results: Dict,
                                iteration: int,
                                market: str,
                                timeframe: str):
        """
        Save iteration results and create visualizations.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Define model path with proper extension
        model_base_path = self.models_dir / f'model_{market}_{timeframe}_iter_{iteration}_{timestamp}'
        
        # Save model
        try:
            results['model'].save(str(model_base_path))
            logger.info(f"Saved model to {model_base_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        # Save scaler
        scaler_path = model_base_path.with_suffix('.scaler.pkl')
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
        
        # Create and save visualizations
        self._create_iteration_plots(results, iteration, market, timeframe, timestamp)
        
        # Update history
        self.history['metrics'].append(self._calculate_metrics(results['backtest_results']))
        self.history['feature_importances'].append(results['model'].feature_importances)
        self.history['model_weights'].append(results['model'].model_weights)

    def _create_iteration_plots(self,
                                 results: Dict,
                                 iteration: int,
                                 market: str,
                                 timeframe: str,
                                 timestamp: str):
        """
        Create detailed plots for the iteration.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
            timestamp (str): Current timestamp.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for symbol, result in results['backtest_results'].items():
            equity_curve = pd.Series(result['equity_curve'])
            ax1.plot(equity_curve.values, label=symbol)
        ax1.set_title(f'Equity Curves - Iteration {iteration + 1}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # 2. Position sizes over time
        ax2 = fig.add_subplot(gs[1, 0])
        for symbol, result in results['backtest_results'].items():
            positions = pd.Series(result['positions'])
            ax2.plot(positions, label=symbol, alpha=0.5)
        ax2.set_title('Position Sizes')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        # 3. Prediction distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for symbol, result in results['backtest_results'].items():
            predictions = pd.Series(result['predictions'])
            sns.histplot(predictions, bins=50, kde=True, label=symbol, ax=ax3, alpha=0.5)
        ax3.set_title('Prediction Distribution')
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # 4. Performance metrics heatmap
        ax4 = fig.add_subplot(gs[2, :])
        metrics = pd.DataFrame(self._calculate_metrics(results['backtest_results'])).T
        if not metrics.empty:
            sns.heatmap(metrics, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
            ax4.set_title('Performance Metrics')
        else:
            logger.warning("No performance metrics available to plot.")
            ax4.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        plot_filename = self.results_dir / f'iteration_{iteration}_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved iteration plot to {plot_filename}")

    def _create_training_summary(self, market: str, timeframe: str):
        """
        Create and save training summary visualizations.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.history['metrics']:
            logger.warning("No training metrics available to create summary.")
            return
        
        # Create summary plots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Performance metrics over iterations
        ax1 = fig.add_subplot(gs[0, :])
        metrics_df = pd.DataFrame([
            {k: np.mean([m[k] for m in metrics.values()])
             for k in next(iter(metrics.values())).keys()}
            for metrics in self.history['metrics']
        ])
        if not metrics_df.empty:
            metrics_df.plot(ax=ax1, marker='o')
            ax1.set_title('Performance Metrics Across Iterations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True)
        else:
            logger.warning("No performance metrics available to plot across iterations.")
            ax1.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Feature importance evolution
        ax2 = fig.add_subplot(gs[1, :])
        if self.history['feature_importances']:
            importance_df = pd.DataFrame(self.history['feature_importances'])
            sns.heatmap(importance_df.T, ax=ax2, cmap='YlOrRd', annot=False)
            ax2.set_title('Feature Importance Evolution')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Feature')
        else:
            logger.warning("No feature importances available to plot.")
            ax2.text(0.5, 0.5, 'No Feature Importances Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. Model weights evolution
        ax3 = fig.add_subplot(gs[2, :])
        if self.history['model_weights']:
            weights_df = pd.DataFrame(self.history['model_weights'])
            weights_df.plot(kind='bar', ax=ax3)
            ax3.set_title('Model Weights Evolution')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Weight')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
        else:
            logger.warning("No model weights available to plot.")
            ax3.text(0.5, 0.5, 'No Model Weights Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        plt.tight_layout()
        summary_plot_filename = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            summary_plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved training summary plot to {summary_plot_filename}")
        
        # Save summary metrics to CSV
        if not self.history['metrics']:
            logger.warning("No training metrics available to save as CSV.")
        else:
            summary_csv_path = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.csv'
            pd.DataFrame(self.history['metrics']).to_csv(summary_csv_path, index=False)
            logger.info(f"Saved training summary metrics to {summary_csv_path}")

    def _generate_comprehensive_report(self, market: str, timeframe: str):
        """
        Generate a comprehensive report in .txt and .csv formats.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_txt_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.txt'
        report_csv_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.csv'
        
        try:
            with open(report_txt_path, 'w') as f:
                f.write(f"Training Report - {market.capitalize()} {timeframe.capitalize()}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for idx, metrics in enumerate(self.history['metrics'], 1):
                    f.write(f"--- Iteration {idx} ---\n")
                    for symbol, symbol_metrics in metrics.items():
                        f.write(f"Symbol: {symbol}\n")
                        for metric_name, value in symbol_metrics.items():
                            if isinstance(value, float):
                                if 'Return' in metric_name or 'Drawdown' in metric_name:
                                    f.write(f"  {metric_name}: {value:.2%}\n")
                                else:
                                    f.write(f"  {metric_name}: {value:.4f}\n")
                            else:
                                f.write(f"  {metric_name}: {value}\n")
                        f.write("\n")
                    f.write(f"Model Weights: {self.history['model_weights'][idx-1]}\n\n")
            logger.info(f"Saved comprehensive report to {report_txt_path}")
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
        
        try:
            # Convert metrics to a single DataFrame
            all_metrics = []
            for idx, metrics in enumerate(self.history['metrics'], 1):
                for symbol, symbol_metrics in metrics.items():
                    row = {'Iteration': idx, 'Symbol': symbol}
                    row.update(symbol_metrics)
                    row.update({
                        'Model Weight_rf': self.history['model_weights'][idx-1].get('rf', np.nan),
                        'Model Weight_gb': self.history['model_weights'][idx-1].get('gb', np.nan),
                        'Model Weight_nn': self.history['model_weights'][idx-1].get('nn', np.nan)
                    })
                    all_metrics.append(row)
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(report_csv_path, index=False)
            logger.info(f"Saved comprehensive report CSV to {report_csv_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

    def _calculate_metrics(self, results: Dict) -> Dict:
        """
        Calculate performance metrics for backtest results.
        
        Args:
            results (Dict): Backtest results for each symbol.
        
        Returns:
            Dict: Performance metrics for each symbol.
        """
        metrics = {}
        for symbol, result in results.items():
            returns = pd.Series(result['returns']).fillna(0)
            if returns.std() == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
            max_drawdown = (1 - (1 + returns).cumprod() / (1 + returns).cumprod().cummax()).max()
            win_rate = (returns > 0).mean()
            profit_factor = abs(returns[returns > 0].sum() / (returns[returns < 0].sum() or 1e-10))
            
            metrics[symbol] = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Profit Factor': profit_factor
            }
        return metrics

    def _run_backtest(self, 
                      model: EnhancedMLModel,
                      data: pd.DataFrame,
                      features: pd.DataFrame,
                      regimes: np.ndarray) -> Dict:
        """
        Run backtest with the trained model using parallel processing.
        
        Args:
            model (EnhancedMLModel): Trained ensemble model.
            data (pd.DataFrame): Prepared data.
            features (pd.DataFrame): Generated feature matrix.
            regimes (np.ndarray): Array indicating market regimes.
        
        Returns:
            Dict: Backtest results for each symbol.
        """
        results = {}
        symbols = data['symbol'].unique()
        
        logger.info("Running backtests in parallel.")
        
        # Prepare data for parallel processing
        symbol_data_list = [data[data['symbol'] == symbol].copy() for symbol in symbols]
        symbol_features_list = [features[data['symbol'] == symbol].copy() for symbol in symbols]
        symbol_regimes_list = [regimes[data['symbol'] == symbol] for symbol in symbols]
        
        # Define backtest function for parallel processing
        def backtest_symbol(symbol, symbol_data, symbol_features, symbol_regimes, model, risk_manager) -> Tuple[str, Dict]:
            """
            Backtest a single symbol.
            
            Args:
                symbol (str): Symbol name.
                symbol_data (pd.DataFrame): Data for the symbol.
                symbol_features (pd.DataFrame): Features for the symbol.
                symbol_regimes (np.ndarray): Regimes for the symbol.
                model (EnhancedMLModel): Trained model.
                risk_manager (RiskManager): Risk manager instance.
            
            Returns:
                Tuple[str, Dict]: Symbol name and its backtest results.
            """
            try:
                logger.info(f"Backtesting symbol: {symbol}")
                
                # Make bulk predictions
                ensemble_pred, confidence = model.predict_bulk(symbol_features, symbol_regimes)
                
                # Calculate positions
                positions = risk_manager.calculate_position_size_bulk(
                    symbol,
                    symbol_data,
                    ensemble_pred,
                    confidence
                )
                
                # Apply stops
                positions, stopped = risk_manager.apply_stops_bulk(
                    symbol,
                    symbol_data['close'],
                    positions
                )
                
                # Calculate returns
                returns = positions.shift() * symbol_data['close'].pct_change()
                
                # Calculate equity curve
                equity_curve = (1 + returns).cumprod()
                
                # Compile results
                symbol_result = {
                    'predictions': ensemble_pred.tolist(),
                    'confidences': confidence.tolist(),
                    'positions': positions.tolist(),
                    'returns': returns.tolist(),
                    'equity_curve': equity_curve.tolist()
                }
                
                logger.info(f"Completed backtest for symbol: {symbol}")
                return (symbol, symbol_result)
            
            except Exception as e:
                logger.error(f"Error backtesting symbol {symbol}: {e}")
                return (symbol, {})
        
        # Parallel processing of backtests
        processed_results = Parallel(n_jobs=-1)(
            delayed(backtest_symbol)(
                symbol, 
                symbol_data, 
                symbol_features, 
                symbol_regimes, 
                model, 
                self.risk_manager
            )
            for symbol, symbol_data, symbol_features, symbol_regimes in zip(symbols, symbol_data_list, symbol_features_list, symbol_regimes_list)
        )
        
        # Aggregate results
        for symbol, res in processed_results:
            if res:
                results[symbol] = res
        
        return results

    def _log_iteration_summary(self, results: Dict, iteration: int):
        """
        Log summary of iteration results.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
        """
        metrics = self._calculate_metrics(results['backtest_results'])
        if not metrics:
            logger.warning("No metrics to summarize for this iteration.")
            return
        avg_metrics = pd.DataFrame(metrics).mean()
        
        logger.info("\nIteration Summary:")
        logger.info(f"Average Sharpe Ratio: {avg_metrics['Sharpe Ratio']:.2f}")
        logger.info(f"Average Annual Return: {avg_metrics['Annual Return']:.2%}")
        logger.info(f"Average Max Drawdown: {avg_metrics['Max Drawdown']:.2%}")
        logger.info(f"Average Win Rate: {avg_metrics['Win Rate']:.2%}")
        logger.info(f"Profit Factor: {avg_metrics['Profit Factor']:.2f}")
        logger.info(f"Model Weights: {results['model'].model_weights}")
        
        # Log cross-validation metrics if available
        if results['cv_metrics']:
            try:
                avg_f1 = {name: np.mean(scores) for name, scores in results['cv_metrics'].items()}
                for name, f1 in avg_f1.items():
                    logger.info(f"Average CV F1 Score for {name.upper()}: {f1:.4f}")
            except Exception as e:
                logger.error(f"Error calculating average CV F1 Score: {str(e)}")

    def _save_iteration_results(self, 
                                results: Dict,
                                iteration: int,
                                market: str,
                                timeframe: str):
        """
        Save iteration results and create visualizations.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Define model path with proper extension
        model_base_path = self.models_dir / f'model_{market}_{timeframe}_iter_{iteration}_{timestamp}'
        
        # Save model
        try:
            results['model'].save(str(model_base_path))
            logger.info(f"Saved model to {model_base_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        # Save scaler
        scaler_path = model_base_path.with_suffix('.scaler.pkl')
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
        
        # Create and save visualizations
        self._create_iteration_plots(results, iteration, market, timeframe, timestamp)
        
        # Update history
        self.history['metrics'].append(self._calculate_metrics(results['backtest_results']))
        self.history['feature_importances'].append(results['model'].feature_importances)
        self.history['model_weights'].append(results['model'].model_weights)

    def _create_iteration_plots(self,
                                 results: Dict,
                                 iteration: int,
                                 market: str,
                                 timeframe: str,
                                 timestamp: str):
        """
        Create detailed plots for the iteration.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
            timestamp (str): Current timestamp.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for symbol, result in results['backtest_results'].items():
            equity_curve = pd.Series(result['equity_curve'])
            ax1.plot(equity_curve.values, label=symbol)
        ax1.set_title(f'Equity Curves - Iteration {iteration + 1}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # 2. Position sizes over time
        ax2 = fig.add_subplot(gs[1, 0])
        for symbol, result in results['backtest_results'].items():
            positions = pd.Series(result['positions'])
            ax2.plot(positions, label=symbol, alpha=0.5)
        ax2.set_title('Position Sizes')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        # 3. Prediction distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for symbol, result in results['backtest_results'].items():
            predictions = pd.Series(result['predictions'])
            sns.histplot(predictions, bins=50, kde=True, label=symbol, ax=ax3, alpha=0.5)
        ax3.set_title('Prediction Distribution')
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # 4. Performance metrics heatmap
        ax4 = fig.add_subplot(gs[2, :])
        metrics = pd.DataFrame(self._calculate_metrics(results['backtest_results'])).T
        if not metrics.empty:
            sns.heatmap(metrics, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
            ax4.set_title('Performance Metrics')
        else:
            logger.warning("No performance metrics available to plot.")
            ax4.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        plot_filename = self.results_dir / f'iteration_{iteration}_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved iteration plot to {plot_filename}")

    def _create_training_summary(self, market: str, timeframe: str):
        """
        Create and save training summary visualizations.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.history['metrics']:
            logger.warning("No training metrics available to create summary.")
            return
        
        # Create summary plots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Performance metrics over iterations
        ax1 = fig.add_subplot(gs[0, :])
        metrics_df = pd.DataFrame([
            {k: np.mean([m[k] for m in metrics.values()])
             for k in next(iter(metrics.values())).keys()}
            for metrics in self.history['metrics']
        ])
        if not metrics_df.empty:
            metrics_df.plot(ax=ax1, marker='o')
            ax1.set_title('Performance Metrics Across Iterations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True)
        else:
            logger.warning("No performance metrics available to plot across iterations.")
            ax1.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Feature importance evolution
        ax2 = fig.add_subplot(gs[1, :])
        if self.history['feature_importances']:
            importance_df = pd.DataFrame(self.history['feature_importances'])
            sns.heatmap(importance_df.T, ax=ax2, cmap='YlOrRd', annot=False)
            ax2.set_title('Feature Importance Evolution')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Feature')
        else:
            logger.warning("No feature importances available to plot.")
            ax2.text(0.5, 0.5, 'No Feature Importances Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. Model weights evolution
        ax3 = fig.add_subplot(gs[2, :])
        if self.history['model_weights']:
            weights_df = pd.DataFrame(self.history['model_weights'])
            weights_df.plot(kind='bar', ax=ax3)
            ax3.set_title('Model Weights Evolution')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Weight')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
        else:
            logger.warning("No model weights available to plot.")
            ax3.text(0.5, 0.5, 'No Model Weights Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        plt.tight_layout()
        summary_plot_filename = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            summary_plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved training summary plot to {summary_plot_filename}")
        
        # Save summary metrics to CSV
        if not self.history['metrics']:
            logger.warning("No training metrics available to save as CSV.")
        else:
            summary_csv_path = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.csv'
            pd.DataFrame(self.history['metrics']).to_csv(summary_csv_path, index=False)
            logger.info(f"Saved training summary metrics to {summary_csv_path}")

    def _generate_comprehensive_report(self, market: str, timeframe: str):
        """
        Generate a comprehensive report in .txt and .csv formats.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_txt_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.txt'
        report_csv_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.csv'
        
        try:
            with open(report_txt_path, 'w') as f:
                f.write(f"Training Report - {market.capitalize()} {timeframe.capitalize()}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for idx, metrics in enumerate(self.history['metrics'], 1):
                    f.write(f"--- Iteration {idx} ---\n")
                    for symbol, symbol_metrics in metrics.items():
                        f.write(f"Symbol: {symbol}\n")
                        for metric_name, value in symbol_metrics.items():
                            if isinstance(value, float):
                                if 'Return' in metric_name or 'Drawdown' in metric_name:
                                    f.write(f"  {metric_name}: {value:.2%}\n")
                                else:
                                    f.write(f"  {metric_name}: {value:.4f}\n")
                            else:
                                f.write(f"  {metric_name}: {value}\n")
                        f.write("\n")
                    f.write(f"Model Weights: {self.history['model_weights'][idx-1]}\n\n")
            logger.info(f"Saved comprehensive report to {report_txt_path}")
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
        
        try:
            # Convert metrics to a single DataFrame
            all_metrics = []
            for idx, metrics in enumerate(self.history['metrics'], 1):
                for symbol, symbol_metrics in metrics.items():
                    row = {'Iteration': idx, 'Symbol': symbol}
                    row.update(symbol_metrics)
                    row.update({
                        'Model Weight_rf': self.history['model_weights'][idx-1].get('rf', np.nan),
                        'Model Weight_gb': self.history['model_weights'][idx-1].get('gb', np.nan),
                        'Model Weight_nn': self.history['model_weights'][idx-1].get('nn', np.nan)
                    })
                    all_metrics.append(row)
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(report_csv_path, index=False)
            logger.info(f"Saved comprehensive report CSV to {report_csv_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

    def _log_iteration_summary(self, results: Dict, iteration: int):
        """
        Log summary of iteration results.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
        """
        metrics = self._calculate_metrics(results['backtest_results'])
        if not metrics:
            logger.warning("No metrics to summarize for this iteration.")
            return
        avg_metrics = pd.DataFrame(metrics).mean()
        
        logger.info("\nIteration Summary:")
        logger.info(f"Average Sharpe Ratio: {avg_metrics['Sharpe Ratio']:.2f}")
        logger.info(f"Average Annual Return: {avg_metrics['Annual Return']:.2%}")
        logger.info(f"Average Max Drawdown: {avg_metrics['Max Drawdown']:.2%}")
        logger.info(f"Average Win Rate: {avg_metrics['Win Rate']:.2%}")
        logger.info(f"Profit Factor: {avg_metrics['Profit Factor']:.2f}")
        logger.info(f"Model Weights: {results['model'].model_weights}")
        
        # Log cross-validation metrics if available
        if results['cv_metrics']:
            try:
                avg_f1 = {name: np.mean(scores) for name, scores in results['cv_metrics'].items()}
                for name, f1 in avg_f1.items():
                    logger.info(f"Average CV F1 Score for {name.upper()}: {f1:.4f}")
            except Exception as e:
                logger.error(f"Error calculating average CV F1 Score: {e}")

    def _create_iteration_plots(self,
                                 results: Dict,
                                 iteration: int,
                                 market: str,
                                 timeframe: str,
                                 timestamp: str):
        """
        Create detailed plots for the iteration.
        
        Args:
            results (Dict): Results from the iteration.
            iteration (int): Current iteration number.
            market (str): Market name.
            timeframe (str): Timeframe name.
            timestamp (str): Current timestamp.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        for symbol, result in results['backtest_results'].items():
            equity_curve = pd.Series(result['equity_curve'])
            ax1.plot(equity_curve.values, label=symbol)
        ax1.set_title(f'Equity Curves - Iteration {iteration + 1}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # 2. Position sizes over time
        ax2 = fig.add_subplot(gs[1, 0])
        for symbol, result in results['backtest_results'].items():
            positions = pd.Series(result['positions'])
            ax2.plot(positions, label=symbol, alpha=0.5)
        ax2.set_title('Position Sizes')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        # 3. Prediction distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for symbol, result in results['backtest_results'].items():
            predictions = pd.Series(result['predictions'])
            sns.histplot(predictions, bins=50, kde=True, label=symbol, ax=ax3, alpha=0.5)
        ax3.set_title('Prediction Distribution')
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Frequency')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
        
        # 4. Performance metrics heatmap
        ax4 = fig.add_subplot(gs[2, :])
        metrics = pd.DataFrame(self._calculate_metrics(results['backtest_results'])).T
        if not metrics.empty:
            sns.heatmap(metrics, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax4)
            ax4.set_title('Performance Metrics')
        else:
            logger.warning("No performance metrics available to plot.")
            ax4.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        plot_filename = self.results_dir / f'iteration_{iteration}_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved iteration plot to {plot_filename}")

    def _create_training_summary(self, market: str, timeframe: str):
        """
        Create and save training summary visualizations.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not self.history['metrics']:
            logger.warning("No training metrics available to create summary.")
            return
        
        # Create summary plots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2)
        
        # 1. Performance metrics over iterations
        ax1 = fig.add_subplot(gs[0, :])
        metrics_df = pd.DataFrame([
            {k: np.mean([m[k] for m in metrics.values()])
             for k in next(iter(metrics.values())).keys()}
            for metrics in self.history['metrics']
        ])
        if not metrics_df.empty:
            metrics_df.plot(ax=ax1, marker='o')
            ax1.set_title('Performance Metrics Across Iterations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True)
        else:
            logger.warning("No performance metrics available to plot across iterations.")
            ax1.text(0.5, 0.5, 'No Metrics Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Feature importance evolution
        ax2 = fig.add_subplot(gs[1, :])
        if self.history['feature_importances']:
            importance_df = pd.DataFrame(self.history['feature_importances'])
            sns.heatmap(importance_df.T, ax=ax2, cmap='YlOrRd', annot=False)
            ax2.set_title('Feature Importance Evolution')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Feature')
        else:
            logger.warning("No feature importances available to plot.")
            ax2.text(0.5, 0.5, 'No Feature Importances Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        # 3. Model weights evolution
        ax3 = fig.add_subplot(gs[2, :])
        if self.history['model_weights']:
            weights_df = pd.DataFrame(self.history['model_weights'])
            weights_df.plot(kind='bar', ax=ax3)
            ax3.set_title('Model Weights Evolution')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Weight')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
        else:
            logger.warning("No model weights available to plot.")
            ax3.text(0.5, 0.5, 'No Model Weights Available', horizontalalignment='center',
                     verticalalignment='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        plt.tight_layout()
        summary_plot_filename = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.png'
        plt.savefig(
            summary_plot_filename,
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        logger.info(f"Saved training summary plot to {summary_plot_filename}")
        
        # Save summary metrics to CSV
        if not self.history['metrics']:
            logger.warning("No training metrics available to save as CSV.")
        else:
            summary_csv_path = self.results_dir / f'training_summary_{market}_{timeframe}_{timestamp}.csv'
            pd.DataFrame(self.history['metrics']).to_csv(summary_csv_path, index=False)
            logger.info(f"Saved training summary metrics to {summary_csv_path}")

    def _generate_comprehensive_report(self, market: str, timeframe: str):
        """
        Generate a comprehensive report in .txt and .csv formats.
        
        Args:
            market (str): Market name.
            timeframe (str): Timeframe name.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_txt_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.txt'
        report_csv_path = self.results_dir / f'training_report_{market}_{timeframe}_{timestamp}.csv'
        
        try:
            with open(report_txt_path, 'w') as f:
                f.write(f"Training Report - {market.capitalize()} {timeframe.capitalize()}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for idx, metrics in enumerate(self.history['metrics'], 1):
                    f.write(f"--- Iteration {idx} ---\n")
                    for symbol, symbol_metrics in metrics.items():
                        f.write(f"Symbol: {symbol}\n")
                        for metric_name, value in symbol_metrics.items():
                            if isinstance(value, float):
                                if 'Return' in metric_name or 'Drawdown' in metric_name:
                                    f.write(f"  {metric_name}: {value:.2%}\n")
                                else:
                                    f.write(f"  {metric_name}: {value:.4f}\n")
                            else:
                                f.write(f"  {metric_name}: {value}\n")
                        f.write("\n")
                    f.write(f"Model Weights: {self.history['model_weights'][idx-1]}\n\n")
            logger.info(f"Saved comprehensive report to {report_txt_path}")
        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
        
        try:
            # Convert metrics to a single DataFrame
            all_metrics = []
            for idx, metrics in enumerate(self.history['metrics'], 1):
                for symbol, symbol_metrics in metrics.items():
                    row = {'Iteration': idx, 'Symbol': symbol}
                    row.update(symbol_metrics)
                    row.update({
                        'Model Weight_rf': self.history['model_weights'][idx-1].get('rf', np.nan),
                        'Model Weight_gb': self.history['model_weights'][idx-1].get('gb', np.nan),
                        'Model Weight_nn': self.history['model_weights'][idx-1].get('nn', np.nan)
                    })
                    all_metrics.append(row)
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(report_csv_path, index=False)
            logger.info(f"Saved comprehensive report CSV to {report_csv_path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")

def main():
    """
    Main execution function.
    """
    parser = ArgumentParser(description='Run progressive training')
    parser.add_argument('--market', type=str, default='stocks', 
                       choices=['stocks', 'crypto'],
                       help='Market to train on')
    parser.add_argument('--timeframe', type=str, default='daily',
                       choices=['daily', 'hourly'],
                       help='Timeframe for analysis')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of training iterations')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run in quick test mode with subset of data')
    
    args = parser.parse_args()
    
    try:
        trainer = ProgressiveTraining(quick_test=args.quick_test)
        trainer.run_progressive_training(
            market=args.market,
            timeframe=args.timeframe,
            n_iterations=args.iterations
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

