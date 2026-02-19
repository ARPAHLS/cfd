"""
Feature engineering module for Credit Card Fraud Detection.
"""
import pandas as pd
from src.utils import logger

def feature_engineering(data):
    """
    Generates new features for the dataset.
    """
    logger.info("Starting feature engineering...")
    
    # 1. Time features
    # 'step' is hours.
    data['hour_of_day'] = data['step'] % 24

    # 2. Transaction Type encoding
    # We use One-Hot Encoding for 'type'
    data = pd.get_dummies(data, columns=['type'], prefix='type', drop_first=True)
    
    # 3. Behavioral Features
    # Error in balance updates (orig)
    # The difference between old - new should be equal to amount (for exact transactions)
    # Fraudsters might manipulate this.
    data['errorBalanceOrig'] = data['newBalanceOrig'] + data['amount'] - data['oldBalanceOrig']

    # Error in balance updates (dest)
    data['errorBalanceDest'] = data['oldBalanceDest'] + data['amount'] - data['newBalanceDest']
    
    # 4. Binary flags
    # isMovement: CASH_OUT or TRANSFER (Simulating the codecademy logic but more robust
    # via dummies)
    # Actually, we already have dummies, so 'type_TRANSFER' and 'type_CASH_OUT' exist.
    
    # Flag for large transaction (dataset has isFlaggedFraud, but let's make our own)
    # data['isLargeTransaction'] = (data['amount'] > 200000).astype(int)
    
    # Drop irrelevant columns
    # nameOrig and nameDest are high cardinality ID strings.
    # In a real system, we might track frequency of these IDs, but for this generic model,
    # we drop them.
    # isFlaggedFraud is a rule-based label in the dataset, we can keep it as a feature
    # or drop it.
    # The goal is to predict 'isFraud'.

    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    logger.info("Feature engineering completed. Features: %s", data.columns.tolist())
    return data
