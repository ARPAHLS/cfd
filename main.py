import argparse
import sys
import os
from src.data_loader import load_data, clean_data
from src.features import feature_engineering
from src.model import FraudDetector
from src.evaluation import evaluate_model
from src.utils import logger

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection System")
    parser.add_argument('--data', type=str, default='data/PS_20174392719_1491204439457_log.csv', help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Mode: train or predict')
    parser.add_argument('--model_type', type=str, choices=['rf', 'xgb'], default='rf', help='Model type: rf (Random Forest) or xgb (XGBoost)')
    parser.add_argument('--model_path', type=str, default='models/fraud_model.pkl', help='Path to save/load model')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            logger.info("Starting training pipeline...")
            
            # 1. Load Data
            df = load_data(args.data)
            
            # 2. Clean Data
            df = clean_data(df)
            
            # 3. Feature Engineering
            df = feature_engineering(df)
            
            # 4. Initialize Model
            detector = FraudDetector(model_type=args.model_type)
            
            # 5. Prepare Data
            X_train_scaled, X_test_scaled, y_train, y_test = detector.prepare_data(df)
            
            # 6. Train Model
            detector.train(X_train_scaled, y_train)
            
            # 7. Evaluate Model
            # We pass the underlying sklearn model to the evaluation function
            evaluate_model(detector.model, X_test_scaled, y_test)
            
            # 8. Save Model
            detector.save_model(args.model_path)
            
            logger.info("Training pipeline completed successfully.")
            
        elif args.mode == 'predict':
            logger.info("Starting prediction pipeline...")
            
            if not os.path.exists(args.model_path):
                logger.error(f"Model path {args.model_path} does not exist. Train first.")
                sys.exit(1)
                
            # Load model
            detector = FraudDetector()
            detector.load_model(args.model_path)
            
            # For prediction, we usually expect a file or single inputs. 
            # For this demo, let's load the data, process it, and run prediction on a sample
            # In a real API, this would take JSON input.
            
            logger.info(f"Loading data from {args.data} for prediction simulation...")
            df = load_data(args.data)
            df = clean_data(df)
            df = feature_engineering(df)
            
            # Let's take a sample of 10 records
            sample = df.sample(10)
            ground_truth = sample['isFraud']
            features = sample.drop(columns=['isFraud'])
            
            logger.info("Running predictions on sample...")
            preds = detector.predict(features)
            probs = detector.predict_proba(features)
            
            for i, (pred, prob, true_val) in enumerate(zip(preds, probs, ground_truth)):
                logger.info(f"Sample {i}: Pred={pred}, ProbFraud={prob[1]:.4f}, Actual={true_val}")
                
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
