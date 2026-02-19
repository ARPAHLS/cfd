import pytest
import pandas as pd
import os
import sys
import glob

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import clean_data
from src.features import feature_engineering
from src.model import FraudDetector

MODEL_PATH = 'models/fraud_model.pkl'
DATA_DIR = 'data'

def get_data_file():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if files:
        return files[0]
    return None

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
@pytest.mark.skipif(get_data_file() is None, reason="No data file found")
def test_model_prediction_on_real_sample():
    """
    Integration test to verify the model works on real data slices
    and produces valid probabilities.
    """
    data_file = get_data_file()
    print(f"Testing on data file: {data_file}")
    
    # Load just a small sample to be fast
    # nrows=1000 to get enough variety
    df = pd.read_csv(data_file, nrows=1000)
    
    # Run full pipeline on this sample
    df = clean_data(df)
    df = feature_engineering(df)
    
    # Separate features and target (if exists)
    if 'isFraud' in df.columns:
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']
    else:
        X = df
        y = None
    
    # Load Model
    detector = FraudDetector()
    detector.load_model(MODEL_PATH)
    
    # Predict
    preds = detector.predict(X)
    probs = detector.predict_proba(X)
    
    # Assertions
    assert len(preds) == len(X)
    assert len(probs) == len(X)
    
    # Check probabilities are valid (0 to 1)
    # probs is usually [n_samples, 2] for binary classification
    assert probs.shape[1] == 2
    assert (probs >= 0).all() and (probs <= 1).all()
    
    # Check if we can detect at least one "Safe" transaction (should be easy)
    # In 1000 rows of PaySim, most are safe (0).
    assert 0 in preds
    
    # Optional: We know y exists if we loaded from csv with isFraud column
    if y is not None and 1 in y.values:
        # Just verifying we can inspect it, logic flow check
        pass
