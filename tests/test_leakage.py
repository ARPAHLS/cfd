"""
Tests for data leakage in feature engineering.
"""
import pandas as pd
import pytest
from src.features import feature_engineering
from src.data_loader import clean_data

@pytest.fixture(name="clean_sample_data")
def fixture_clean_sample_data():
    """
    Fixture that loads and cleans a small sample of data.
    """
    try:
        # Try to load real data if available, else fallback to dummy
        # Using a small chunk to keep tests fast
        data = pd.read_csv('data/PS_20174392719_1491204439457_log.csv', nrows=1000)
    except FileNotFoundError:
        data = pd.read_csv('data/dummy.csv')

    return clean_data(data)

def test_no_single_feature_leakage(clean_sample_data):
    """
    Ensures that no single feature has a 1.0 correlation with the target.
    A correlation of 1.0 (or -1.0) often indicates data leakage (e.g. label included in features).
    """
    # Create features
    df_features = feature_engineering(clean_sample_data.copy())

    # Calculate correlations
    if 'isFraud' not in df_features.columns:
        pytest.skip("Target column 'isFraud' missing from data")

    correlations = df_features.corr()['isFraud'].drop('isFraud')

    # Check if any feature has perfect correlation
    # We use 0.99 as a threshold to be safe, though 1.0 is the strict definition of a leak
    leaks = correlations[abs(correlations) >= 0.99]

    assert len(leaks) == 0, f"Potential data leakage detected in features: {leaks.index.tolist()}"

def test_error_balance_not_perfect_predictor(clean_sample_data):
    """
    Specific check for 'errorBalanceOrig' which was suspected of leakage.
    """
    df_features = feature_engineering(clean_sample_data.copy())

    # It shouldn't be perfectly correlated
    corr = df_features['errorBalanceOrig'].corr(df_features['isFraud'])
    assert abs(corr) < 0.99, f"errorBalanceOrig is too highly correlated: {corr}"
