"""
Model training and prediction module.
"""
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils import logger

class FraudDetector:
    """
    Wrapper class for fraud detection models.
    """
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def prepare_data(self, data, target_col='isFraud', test_size=0.2):
        """
        Prepares data for training/testing by splitting and scaling.
        """
        logger.info("Preparing data for training...")
        
        X = data.drop(columns=[target_col]) 
        y = data[target_col] 
        
        # Save feature names for inference alignment
        self.feature_names = X.columns.tolist()
        
        logger.info("Features shape: %s, Target shape: %s", X.shape, y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        logger.info("Scaling features...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train) 
        X_test_scaled = self.scaler.transform(X_test) 
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train): 
        """
        Trains the model.
        """
        logger.info("Training %s model...", self.model_type)
        if self.model_type == 'rf':
            # random forest with balanced class weights
            self.model = RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=42, n_jobs=-1, verbose=1
            )
        elif self.model_type == 'xgb':
            # XGBoost
            # Calculate scale_pos_weight for imbalance
            try:
                # y_train might be a Series, connect to numpy
                y_np = y_train.values if hasattr(y_train, 'values') else y_train
                num_neg = np.sum(y_np == 0)
                num_pos = np.sum(y_np == 1)
                ratio = float(num_neg) / float(num_pos) if num_pos > 0 else 1.0
                logger.info("XGBoost scale_pos_weight: %s", ratio)
                
                self.model = XGBClassifier(
                    scale_pos_weight=ratio, n_jobs=-1, random_state=42, eval_metric='logloss'
                )
            except Exception as e: 
                logger.error("Error configuring XGBoost: %s", e)
                raise e
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        if self.model is None:
            raise ValueError("Model initialization failed.")

        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

    def _align_features(self, X):
        """Aligns input features to match training features."""
        if self.feature_names is None:
            return X
        
        # Reindex ensures all training columns exist (filled with 0) and drops extras
        return X.reindex(columns=self.feature_names, fill_value=0)

    def predict(self, X): 
        """Predicts class labels."""
        X = self._align_features(X)
        X_scaled = self.scaler.transform(X) 
        return self.model.predict(X_scaled)

    def predict_proba(self, X): 
        """Predicts class probabilities."""
        X = self._align_features(X)
        X_scaled = self.scaler.transform(X) 
        return self.model.predict_proba(X_scaled)

    def save_model(self, filepath='models/fraud_model.pkl'):
        """Saves the trained model to disk."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        payload = {
            'model': self.model,
            'scaler': self.scaler,
            'type': self.model_type,
            'feature_names': self.feature_names
        }
        joblib.dump(payload, filepath)
        logger.info("Model saved to %s", filepath)

    def load_model(self, filepath='models/fraud_model.pkl'):
        """Loads a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data.get('type', 'rf')
        self.feature_names = data.get('feature_names', None)
        logger.info("Model loaded from %s", filepath)
