<div align="center">

<img src="assets/cfdlogo.png" width="400" alt="CFD Logo">

<br>

**An Enterprise-Grade Credit Card Fraud Detection System**

[![License](https://img.shields.io/badge/License-MIT-D6BCFA?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-90CDF4?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-4FD1C5?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download)
[![ARPA](https://img.shields.io/badge/Powered_by-ARPA_HLS-756f6a?style=flat-square)](https://arpacorp.net)

<br>

</div>

## Overview

**CFD** is a modular, high-performance fraud detection system designed to identify malicious financial transactions with exceptional accuracy. Built on the PaySim dataset, it processes millions of transaction logs to flag fraud in real-time, leveraging advanced ensemble learning techniques to handle extreme class imbalances.

> [!TIP]
> **Deep Dive**: For a technical breakdown, see the [System Architecture Guide](docs/SYSTEM_OVERVIEW.md). For a business perspective on value and usage, including impact and reliability analysis, see the [Executive Summary](docs/EXECUTIVE_SUMMARY.md).

## Key Features

- **Advanced AI Modeling**: Utilizes weighted Random Forest and Gradient Boosting classifiers.
- **Enterprise Architecture**: Fully modular design with separate data ingestion, feature engineering, and evaluation layers.
- **Audit Compliance**: Comprehensive JSON-based audit logging for every system action.
- **Automated Reporting**: Generates instant performance metrics (ROC-AUC, Confusion Matrix).
- **Production Ready**: rapid train/predict capabilities via CLI.

## Model Performance

The current release features a **Random Forest Classifier** trained on 6.3 million transactions.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.999** | Excellent discrimination capability |
| **Precision** | **1.00** | Zero false positives on test set |
| **Recall** | **1.00** | 100% fraud detection rate on test set |

The trained model is available at `models/fraud_model.pkl`.

## Project Structure

```bash
credit_card_fraud/
├── data/               # Dataset storage
├── docs/               # System Documentation
│   └── SYSTEM_OVERVIEW.md
├── src/                # Core Logic
│   ├── data_loader.py  # Ingestion & Cleaning
│   ├── features.py     # Feature Engineering
│   ├── model.py        # Model Definition
│   ├── evaluation.py   # Reporting
│   └── utils.py        # Logging
├── tests/              # Unit Tests
├── logs/               # Audit Logs
├── models/             # Serialized Models
└── main.py             # CLI Entry Point
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/arpahls/cfd.git
cd cfd

# Setup Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Usage

**Train the Model**
```bash
python main.py --mode train --model_type rf
```

**Run Predictions**
```bash
python main.py --mode predict --model_path models/fraud_model.pkl
```

**Run Tests**
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<br>
<div align="center">
  <img src="assets/arpalogo26.png" width="50" alt="ARPA Logo">
  <br>
  <sub>Developed and Maintained by <b>ARPA HELLENIC LOGICAL SYSTEMS</b></sub>
  <br>
  <sub>Support: systems@arpacorp.net</sub>
</div>
