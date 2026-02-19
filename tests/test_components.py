"""
Unit tests for individual components of the pipeline.
"""
import pytest
import pandas as pd
from src.data_loader import clean_data
from src.features import feature_engineering
from src.model import FraudDetector

@pytest.fixture
def sample_data():
    """
    Creates a sample dataframe for testing.
    """
    data = {
        'step': [1] * 10,
        'type': [
            'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER',
            'CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT'
        ],
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

def test_clean_data(sample_data): # pylint: disable=redefined-outer-name
    """
    Test data cleaning function.
    """
    clean_df = clean_data(sample_data)
    # Check renaming
    assert 'oldBalanceOrig' in clean_df.columns
    assert 'oldbalanceOrg' not in clean_df.columns

def test_feature_engineering(sample_data): # pylint: disable=redefined-outer-name
    """
    Test feature engineering function.
    """
    data = clean_data(sample_data)
    feat_df = feature_engineering(data)

    # Check new features
    assert 'hour_of_day' in feat_df.columns
    assert 'errorBalanceOrig' in feat_df.columns
    assert 'type_TRANSFER' in feat_df.columns

    # Check dropped columns
    assert 'nameOrig' not in feat_df.columns
    assert 'isFlaggedFraud' not in feat_df.columns

def test_model_training(sample_data): # pylint: disable=redefined-outer-name
    """
    Test model training execution.
    """
    data = clean_data(sample_data)
    data = feature_engineering(data)

    detector = FraudDetector(model_type='rf')
    train_feat, test_feat, train_labels, _ = detector.prepare_data(data, test_size=0.33)

    detector.train(train_feat, train_labels)

    assert detector.model is not None

    # We use detector.model.predict directly because X_test
    # from prepare_data is already scaled
    preds = detector.model.predict(test_feat)
    assert len(preds) == len(data) - len(train_labels) # Check lengths match test set
