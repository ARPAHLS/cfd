import pytest
import pandas as pd
import numpy as np
import os
from src.data_loader import clean_data
from src.features import feature_engineering
from src.model import FraudDetector

@pytest.fixture
def sample_df():
    data = {
        'step': [1] * 10,
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT'],
        'amount': [100.0, 200.0, 300.0, 150.0, 250.0, 350.0, 100.0, 200.0, 300.0, 400.0],
        'nameOrig': [f'C{i}' for i in range(10)],
        'oldbalanceOrg': [1000.0 + i*100 for i in range(10)],
        'newbalanceOrig': [900.0 + i*100 for i in range(10)],
        'nameDest': [f'M{i}' for i in range(10)],
        'oldbalanceDest': [0.0 + i*10 for i in range(10)],
        'newbalanceDest': [0.0 + i*20 for i in range(10)],
        'isFraud': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # 2 Frauds, 8 Non-Frauds
        'isFlaggedFraud': [0] * 10
    }
    return pd.DataFrame(data)

def test_clean_data(sample_df):
    clean_df = clean_data(sample_df)
    # Check renaming
    assert 'oldBalanceOrig' in clean_df.columns
    assert 'oldbalanceOrg' not in clean_df.columns
    
def test_feature_engineering(sample_df):
    df = clean_data(sample_df)
    feat_df = feature_engineering(df)
    
    # Check new features
    assert 'hour_of_day' in feat_df.columns
    assert 'errorBalanceOrig' in feat_df.columns
    assert 'type_TRANSFER' in feat_df.columns
    
    # Check dropped columns
    assert 'nameOrig' not in feat_df.columns
    assert 'isFlaggedFraud' not in feat_df.columns
    
def test_model_training(sample_df):
    df = clean_data(sample_df)
    df = feature_engineering(df)
    
    detector = FraudDetector(model_type='rf')
    X_train, X_test, y_train, y_test = detector.prepare_data(df, test_size=0.33)
    
    detector.train(X_train, y_train)
    
    assert detector.model is not None
    
    preds = detector.predict(X_test) # Note: predict expects raw features if calling wrapper? 
    # Wait, my wrapper 'predict' does: X_scaled = self.scaler.transform(X)
    # But in test_model_training I got X_test which is ALREADY scaled from prepare_data.
    # So I should not use detector.predict(X_test) from prepare_data output.
    # I should use detector.model.predict(X_test)
    
    preds = detector.model.predict(X_test)
    assert len(preds) == len(y_test)
