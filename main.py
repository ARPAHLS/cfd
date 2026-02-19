
"""
Main execution script for the Credit Card Fraud Detection System.
"""
import argparse
import sys
import os
from src.data_loader import load_data, clean_data
from src.features import feature_engineering
from src.model import FraudDetector
from src.evaluation import evaluate_model
from src.utils import logger

def run_training(args):
    """
    Executes the training pipeline.
    """
    logger.info("Starting training pipeline...")

    # 1. Load Data
    data = load_data(args.data)

    # 2. Clean Data
    data = clean_data(data)

    # 3. Feature Engineering
    data = feature_engineering(data)

    # 4. Initialize Model
    detector = FraudDetector(model_type=args.model_type)

    # 5. Prepare Data
    # pylint: disable=unbalanced-tuple-unpacking
    train_features, test_features, train_labels, test_labels = detector.prepare_data(data)

    # 6. Train Model
    detector.train(train_features, train_labels)

    # 7. Evaluate Model
    evaluate_model(detector.model, test_features, test_labels)

    # 8. Save Model
    detector.save_model(args.model_path)

    logger.info("Training pipeline completed successfully.")

def run_prediction(args):
    """
    Executes the prediction pipeline.
    """
    logger.info("Starting prediction pipeline...")

    if not os.path.exists(args.model_path):
        logger.error("Model path %s does not exist. Train first.", args.model_path)
        sys.exit(1)

    # Load model
    detector = FraudDetector()
    detector.load_model(args.model_path)

    # For prediction, we usually expect a file or single inputs.
    # For this demo, we load data, process it, and run prediction on a sample
    
    logger.info("Loading data from %s for prediction simulation...", args.data)
    data = load_data(args.data)
    data = clean_data(data)
    data = feature_engineering(data)

    # Let's take a sample of 10 records
    sample = data.sample(10)
    ground_truth = sample['isFraud']
    features = sample.drop(columns=['isFraud'])

    logger.info("Running predictions on sample...")
    preds = detector.predict(features)
    probs = detector.predict_proba(features)

    for index, (pred, prob, true_val) in enumerate(zip(preds, probs, ground_truth)):
        logger.info(
            "Sample %s: Pred=%s, ProbFraud=%.4f, Actual=%s",
            index, pred, prob[1], true_val
        )

def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection System")
    parser.add_argument(
        '--data', type=str, default='data/PS_20174392719_1491204439457_log.csv',
        help='Path to dataset'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'predict'], default='train',
        help='Mode: train or predict'
    )
    parser.add_argument(
        '--model_type', type=str, choices=['rf', 'xgb'], default='rf',
        help='Model type: rf (Random Forest) or xgb (XGBoost)'
    )
    parser.add_argument(
        '--model_path', type=str, default='models/fraud_model.pkl',
        help='Path to save/load model'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'train':
            run_training(args)
        elif args.mode == 'predict':
            run_prediction(args)

    except Exception as exc: # pylint: disable=broad-except
        logger.error("An error occurred: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
