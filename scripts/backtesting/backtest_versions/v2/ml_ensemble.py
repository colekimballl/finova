# ml_ensemble.py

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import tensorflow as tf

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Check TensorFlow availability
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow is not installed. Neural Network will not be available.")

class EnhancedMLModel:
    def __init__(self, config: Dict):
        """
        Initialize the EnhancedMLModel with ensemble of Random Forest, Gradient Boosting, and Neural Network.
        
        Args:
            config (Dict): Configuration dictionary containing model parameters.
        """
        self.config = config
        self.models = {
            'rf': RandomForestClassifier(**self.config.get('rf_params', {})),
            'gb': GradientBoostingClassifier(**self.config.get('gb_params', {}))
        }
        
        if TENSORFLOW_AVAILABLE:
            logger.info("Adding Neural Network to ensemble.")
            self.models['nn'] = self._create_neural_net()
        
        # Initialize feature importances and model weights
        self.feature_importances = {}
        self.model_weights = self._initialize_weights()
    
    def _create_neural_net(self) -> tf.keras.Model:
        """
        Create and compile the Neural Network model.
        
        Returns:
            tf.keras.Model: Compiled Neural Network model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.config['input_dim'],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.get('nn_params', {}).get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _initialize_weights(self) -> Dict[str, float]:
        """
        Initialize model weights for the ensemble. Default weights are equal.
        
        Returns:
            Dict[str, float]: Dictionary of model weights.
        """
        models = list(self.models.keys())
        weight = 1.0 / len(models)
        weights = {model: weight for model in models}
        logger.info(f"Initialized model weights: {weights}")
        return weights
    
    def train(self, X: pd.DataFrame, y: pd.Series, regimes: np.ndarray) -> Dict[str, List[float]]:
        """
        Train each model in the ensemble using Stratified K-Fold cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            regimes (np.ndarray): Array indicating market regimes.
        
        Returns:
            Dict[str, List[float]]: Dictionary containing F1 scores for each model across folds.
        """
        logger.info("Starting training of ensemble models.")
        skf = StratifiedKFold(n_splits=self.config.get('cv_folds', 5), shuffle=True, random_state=42)
        cv_metrics = {model: [] for model in self.models.keys()}
        
        for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
            logger.info(f"Starting Fold {fold}")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            regimes_train = regimes[train_index]
            regimes_val = regimes[val_index]
            
            for name, model in self.models.items():
                logger.info(f"Training {name.upper()} model.")
                if name != 'nn':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    f1 = f1_score(y_val, y_pred)
                    cv_metrics[name].append(f1)
                    logger.info(f"Fold {fold} - {name.upper()} F1 Score: {f1:.4f}")
                    
                    # Capture feature importances
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importances.setdefault(name, []).append(model.feature_importances_)
                else:
                    # Neural Network training
                    history = model.fit(
                        X_train, y_train,
                        epochs=self.config['nn_params'].get('epochs', 10),
                        batch_size=self.config['nn_params'].get('batch_size', 32),
                        validation_data=(X_val, y_val),
                        verbose=0
                    )
                    y_pred_prob = model.predict(X_val).flatten()
                    y_pred = (y_pred_prob >= 0.5).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    cv_metrics[name].append(f1)
                    logger.info(f"Fold {fold} - NN F1 Score: {f1:.4f}")
        
        # Calculate average F1 scores
        for name, scores in cv_metrics.items():
            avg_f1 = np.mean(scores)
            logger.info(f"Average CV F1 Score for {name.upper()}: {avg_f1:.4f}")
        
        # Calculate feature importances
        for name in self.feature_importances:
            self.feature_importances[name] = np.mean(self.feature_importances[name], axis=0)
        
        return cv_metrics
    
    def predict_single_model(self, model, X: pd.DataFrame, name: str) -> np.ndarray:
        """
        Make predictions using a single model.
        
        Args:
            model: Trained model.
            X (pd.DataFrame): Feature matrix.
            name (str): Model name.
        
        Returns:
            np.ndarray: Prediction probabilities.
        """
        if name != 'nn':
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X).flatten()
    
    def predict_bulk(self, X: pd.DataFrame, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions with confidence scores in bulk.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            regimes (np.ndarray): Array indicating market regimes.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Ensemble predictions and confidence scores.
        """
        predictions = []
        
        for name, model in self.models.items():
            logger.info(f"Making predictions with {name.upper()} model.")
            pred = self.predict_single_model(model, X, name)
            predictions.append(pred * self.model_weights[name])
        
        ensemble_pred = np.sum(predictions, axis=0)
        confidence = 1 - np.std(predictions, axis=0)
        
        return ensemble_pred, confidence
    
    def save(self, path: str):
        """
        Save all models and additional data (model weights and feature importances).
        
        Args:
            path (str): Directory path to save models.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save sklearn models and Neural Network
        for name, model in self.models.items():
            if name != 'nn':
                model_filepath = os.path.join(path, f"{name}_model.joblib")
                try:
                    joblib.dump(model, model_filepath)
                    logger.info(f"Saved {name.upper()} model to {model_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save {name.upper()} model: {e}")
            elif TENSORFLOW_AVAILABLE:
                model_filepath = os.path.join(path, f"{name}_model.h5")
                try:
                    model.save(model_filepath)
                    logger.info(f"Saved {name.upper()} model to {model_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save {name.upper()} model: {e}")
        
        # Save model weights and feature importances
        try:
            np.save(os.path.join(path, "model_weights.npy"), self.model_weights)
            pd.to_pickle(self.feature_importances, os.path.join(path, "feature_importances.pkl"))
            logger.info("Saved model weights and feature importances.")
        except Exception as e:
            logger.error(f"Failed to save additional data: {e}")

