# feature_generator.py
import pandas as pd
import numpy as np
from typing import List, Dict
import ta
import talib
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.feature_weights = {}
        self.feature_names = []
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for all symbols"""
        logger.info("Generating features...")
        features_df = pd.DataFrame(index=df.index)
        
        # Process each symbol separately
        for symbol in tqdm(df['symbol'].unique(), desc="Processing symbols"):
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_features = self._generate_symbol_features(symbol_data)
            
            # Add features to main DataFrame
            for col in symbol_features.columns:
                features_df.loc[symbol_data.index, col] = symbol_features[col]
        
        # Add 'symbol' as a new column
        features_df['symbol'] = df['symbol']
        
        # Set 'symbol' as an additional index level
        features_df.set_index('symbol', append=True, inplace=True)
        
        # Debug: Verify index levels
        logger.debug(f"Features DataFrame Index: {features_df.index.names}")
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Randomize and apply feature weights
        self.randomize_weights()
        self._apply_weights(features_df)
        
        logger.info(f"Generated {len(self.feature_names)} features")
        return features_df
    
    def _generate_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features for a single symbol"""
        features = pd.DataFrame(index=df.index)
        
        # Base price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Verify 'returns' column exists
        if 'returns' not in features.columns:
            logger.error("Returns column was not created successfully.")
        
        # Technical indicators
        self._add_price_features(df, features)
        self._add_volume_features(df, features)
        self._add_trend_features(df, features)
        self._add_momentum_features(df, features)
        self._add_volatility_features(df, features)
        self._add_pattern_features(df, features)
        
        return features
    
    def _add_price_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add price-based features"""
        # Price levels
        features['price_level'] = df['close'] / df['close'].rolling(20).mean()
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            features[f'distance_to_sma_{period}'] = (df['close'] / features[f'sma_{period}'] - 1)
    
    def _add_volume_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add volume-based features"""
        # Volume indicators
        features['volume_sma_5'] = df['volume'].rolling(5).mean()
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Money flow
        features['money_flow'] = df['volume'] * (df['close'] - df['close'].shift(1))
        features['money_flow_strength'] = features['money_flow'].rolling(20).mean()
    
    def _add_trend_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add trend indicators"""
        # ADX
        features['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # MACD
        macd = ta.trend.macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = ta.trend.macd_signal(df['close'])
        features['macd_diff'] = ta.trend.macd_diff(df['close'])
        
        # Bollinger Bands
        features['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        features['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        features['bb_percent'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    def _add_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add momentum indicators"""
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # Stochastic
        features['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        features['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # ROC
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ta.momentum.roc(df['close'], window=period)
    
    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add volatility indicators"""
        # Standard deviation of returns
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
        
        # Average True Range
        features['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Historical volatility
        features['hist_vol'] = np.log(df['close'] / df['close'].shift(1)).rolling(20).std() * np.sqrt(252)
    
    def _add_pattern_features(self, df: pd.DataFrame, features: pd.DataFrame):
        """Add candlestick pattern indicators using ta-lib"""
        # Basic patterns using ta-lib
        features['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        features['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        features['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Convert pattern indicators to binary (1 if pattern detected, 0 otherwise)
        features['doji'] = features['doji'].apply(lambda x: 1 if x != 0 else 0)
        features['hammer'] = features['hammer'].apply(lambda x: 1 if x != 0 else 0)
        features['shooting_star'] = features['shooting_star'].apply(lambda x: 1 if x != 0 else 0)
    
    def randomize_weights(self):
        """Randomize feature weights"""
        weights = np.random.uniform(0.1, 1.0, len(self.feature_names))
        self.feature_weights = dict(zip(self.feature_names, weights))
        logger.info("Randomized feature weights")
    
    def _apply_weights(self, features: pd.DataFrame):
        """Apply weights to features"""
        for col in features.columns:
            if col in self.feature_weights:
                features[col] = features[col] * self.feature_weights[col]
        
        # Fill NaN values using forward fill and then fill remaining NaNs with 0
        features = features.ffill().fillna(0)
    
    def get_feature_importance(self) -> pd.Series:
        """Get current feature weights"""
        return pd.Series(self.feature_weights)
    
    def print_top_features(self, n: int = 10):
        """Print top N features by weight"""
        weights = self.get_feature_importance()
        print("\nTop Features:")
        print(weights.nlargest(n))

