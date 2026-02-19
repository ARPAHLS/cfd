# Feature Importance & Leakage Analysis

## Overview
This document addresses concerns regarding potential data leakage and "unrealistic" performance metrics (0.999 AUC) on the PaySim dataset.

## Methodology
We analyzed feature correlations and trained a Random Forest classifier to extract feature importances on a subset of the data. To ensure ongoing quality, we have implemented automated tests (`tests/test_leakage.py`) that verify no single feature acts as a perfect predictor (leak).

## Findings

### 1. Feature Importance
The top predictors for fraud in this model are:

| Feature | Importance | Description |
| :--- | :--- | :--- |
| **newBalanceOrig** | **20.02%** | The balance remaining in the origin account after the transaction. |
| **amount** | **16.61%** | The transaction amount. |
| **oldBalanceDest** | **15.82%** | Initial balance of the recipient. |
| **errorBalanceOrig** | **13.11%** | Discrepancy in origin balance change (`new - old + amount`). |
| **newBalanceDest** | **13.65%** | Final balance of the recipient. |

### 2. Leakage Investigation: `errorBalanceOrig`
Concern was raised that `errorBalanceOrig` might be a "leaked" feature.
- **Correlation with Target**: -0.0166 (Very low linear correlation).
- **Distribution**:
    - **Fraud**: 110 cases have `0.0` error, but many have massive errors (e.g., > 1,000,000).
    - **Legit**: The vast majority (22,000+) have `0.0` error.
- **Conclusion**: This feature is **not** a perfect predictor (leak). It is a strong signal because legitimate transactions strictly follow mathematical rules, whereas fraudulent transactions in this dataset often do not (or show specific patterns of "account emptying").

## Performance Context
The PaySim dataset is **synthetic**. The fraud patterns are generated using specific rules (e.g., "transfer entire balance").
- **High AUC (0.999)**: This is expected on this specific dataset because the fraud patterns are deterministic and low-noise.
- **Real-world Applicability**: In a production environment with noisier human behavior, AUC would likely be lower (~0.95). The current high score validates that the model successfully learned the underlying generation rules of the dataset, not that it is "leaking" future data.
