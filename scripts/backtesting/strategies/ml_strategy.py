# strategies/ml_strategy.py

import backtrader as bt
import joblib
import pandas as pd

class MLStrategy(bt.Strategy):
    params = (
        ('model_path', 'models/buy_sell_model.pkl'),
    )

    def __init__(self):
        self.model = joblib.load(self.params.model_path)
        self.dataclose = self.datas[0].close

    def next(self):
        # Extract features for the current timestep
        features = self.extract_features()
        features_df = pd.DataFrame([features])

        # Make prediction
        prediction = self.model.predict(features_df)[0]

        if prediction == 1:
            self.buy()
        elif prediction == -1:
            self.sell()

    def extract_features(self):
        # Example feature extraction
        return {
            'close': self.dataclose[0],
            'sma20': bt.ind.SMA(self.data.close, period=20)[0],
            'rsi14': bt.ind.RSI_SMA(self.data.close, period=14)[0],
            # Add more features as needed
        }

