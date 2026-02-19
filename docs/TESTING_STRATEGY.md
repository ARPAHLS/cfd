# Testing Strategy & Guide

This document outlines the testing framework for the Credit Card Fraud Detection (CFD) system. We employ a dual-layer testing strategy to ensure both code correctness and model performance.

## 1. Unit Tests (`tests/test_components.py`)

**Purpose**: To verify that the individual software components function correctly in isolation.

**What it covers**:
*   **Data Cleaning**: Ensures column renaming logic (`oldbalanceOrg` -> `oldBalanceOrig`) covers all edge cases.
*   **Feature Engineering**: Verifies that mathematical transformations (like `errorBalanceOrig`) and feature extraction (`hour_of_day`) produce the expected numerical output.
*   **Model Wrapper**: Checks that the `FraudDetector` class can initialize, train, and predict on synthetic dummy data without raising software errors.

**When to run**: continuously during development (e.g., after modifying `src/features.py`).

```bash
pytest tests/test_components.py
```

## 2. Integration Tests (`tests/test_integration.py`)

**Purpose**: To verify that the *trained model* behaves correctly when presented with *real* production-like data.

**What it covers**:
*   **Pipeline Integrity**: Loads the actual saved model (`models/fraud_model.pkl`) and feeds it real data samples from `data/`.
*   **Probability Validity**: Asserts that the model outputs valid probability scores (0.0 to 1.0) for every transaction.
*   **End-to-End Execution**: Validates the entire flow from Loading -> Cleaning -> Feature Engineering -> Prediction in a single run.

**When to run**: Before deployment or after retraining the model.

```bash
pytest tests/test_integration.py
```

## Running All Tests

To execute the full test suite:

```bash
pytest tests/
```
