import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
from src.utils import logger

class FraudDetector:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_data(self, df, target_col='isFraud', test_size=0.2):
        logger.info("Preparing data for training...")
        # Drop columns that are not features (like isFlaggedFraud if handled in features.py, 
        # but features.py already dropped non-features. 
        # Just ensure target is separated.)
        
        # Ensure we don't have object columns that weren't encoded
        # (features.py should have handled this, but good to check)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        
        logger.info("Scaling features...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        logger.info(f"Training {self.model_type} model...")
        if self.model_type == 'rf':
            # random forest with balanced class weights
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1, verbose=1)
        elif self.model_type == 'xgb':
            # XGBoost
            # Calculate scale_pos_weight for imbalance
            try:
                # y_train might be a Series, connect to numpy
                y_np = y_train.values if hasattr(y_train, 'values') else y_train
                num_neg = np.sum(y_np == 0)
                num_pos = np.sum(y_np == 1)
                ratio = float(num_neg) / float(num_pos) if num_pos > 0 else 1.0
                logger.info(f"XGBoost scale_pos_weight: {ratio}")
                
                self.model = XGBClassifier(scale_pos_weight=ratio, n_jobs=-1, random_state=42, eval_metric='logloss')
            except Exception as e:
                logger.error(f"Error configuring XGBoost: {e}")
                raise e
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        if self.model is None:
             raise ValueError("Model initialization failed.")

        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save_model(self, filepath='models/fraud_model.pkl'):
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'type': self.model_type}, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/fraud_model.pkl'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data.get('type', 'rf')
        logger.info(f"Model loaded from {filepath}")
