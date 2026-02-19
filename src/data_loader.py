import pandas as pd
import os
from src.utils import logger

def load_data(filepath):
    """
    Loads the credit card fraud dataset.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def clean_data(df):
    """
    Basic data cleaning.
    """
    logger.info("Starting data cleaning...")
    
    # Check for nulls
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logger.warning(f"Found {null_counts} null values. Dropping...")
        df = df.dropna()
    
    # Rename columns for consistency if needed? 
    # The dataset has mixed naming: nameOrig, oldbalanceOrg (missing i), newbalanceOrig
    # Let's clean that up.
    rename_map = {
        'oldbalanceOrg': 'oldBalanceOrig',
        'newbalanceOrig': 'newBalanceOrig',
        'oldbalanceDest': 'oldBalanceDest',
        'newbalanceDest': 'newBalanceDest'
    }
    df = df.rename(columns=rename_map)
    
    logger.info("Data cleaning completed.")
    return df
