import pandas as pd
import numpy as np
from src.utils import logger

def feature_engineering(df):
    """
    Generates new features for the dataset.
    """
    logger.info("Starting feature engineering...")
    
    # 1. Time features
    # 'step' is hours. 
    df['hour_of_day'] = df['step'] % 24
    
    # 2. Transaction Type encoding
    # We use One-Hot Encoding for 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
    
    # 3. Behavioral Features
    # Error in balance updates (orig)
    # The difference between old - new should be equal to amount (for exact transactions)
    # Fraudsters might manipulate this.
    df['errorBalanceOrig'] = df['newBalanceOrig'] + df['amount'] - df['oldBalanceOrig']
    
    # Error in balance updates (dest)
    df['errorBalanceDest'] = df['oldBalanceDest'] + df['amount'] - df['newBalanceDest']
    
    # 4. Binary flags
    # isMovement: CASH_OUT or TRANSFER (Simulating the codecademy logic but more robust via dummies)
    # Actually, we already have dummies, so 'type_TRANSFER' and 'type_CASH_OUT' exist.
    
    # Flag for large transaction (dataset has isFlaggedFraud, but let's make our own)
    # df['isLargeTransaction'] = (df['amount'] > 200000).astype(int) 
    
    # Drop irrelevant columns
    # nameOrig and nameDest are high cardinality ID strings. 
    # In a real system, we might track frequency of these IDs, but for this generic model, we drop them.
    # isFlaggedFraud is a rule-based label in the dataset, we can keep it as a feature or drop it.
    # The goal is to predict 'isFraud'.
    
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    logger.info(f"Feature engineering completed. Features: {df.columns.tolist()}")
    return df
