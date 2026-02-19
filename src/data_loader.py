"""
Data loading and cleaning module.
"""
import os
import pandas as pd
from src.utils import logger

def load_data(filepath):
    """
    Loads the credit card fraud dataset.
    """
    if not os.path.exists(filepath):
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info("Loading data from %s...", filepath)
    try:
        data = pd.read_csv(filepath)
        logger.info("Data loaded successfully. Shape: %s", data.shape)
        return data
    except Exception as exc: # pylint: disable=broad-except
        logger.error("Error loading data: %s", exc)
        raise

def clean_data(data):
    """
    Basic data cleaning.
    """
    logger.info("Starting data cleaning...")
    
    # Check for nulls
    null_counts = data.isnull().sum().sum()
    if null_counts > 0:
        logger.warning("Found %s null values. Dropping...", null_counts)
        data = data.dropna()

    # Rename columns for consistency if needed?
    # The dataset has mixed naming: nameOrig, oldbalanceOrg (missing i), newbalanceOrig
    # Let's clean that up.
    rename_map = {
        'oldbalanceOrg': 'oldBalanceOrig',
        'newbalanceOrig': 'newBalanceOrig',
        'oldbalanceDest': 'oldBalanceDest',
        'newbalanceDest': 'newBalanceDest'
    }
    data = data.rename(columns=rename_map)

    logger.info("Data cleaning completed.")
    return data
