from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import logger

def evaluate_model(model, X_test_scaled, y_test, report_dir='reports'):
    """
    Evaluates the model and generates reports.
    """
    logger.info("Evaluating model...")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    y_pred = model.predict(X_test_scaled) # Model object here is the underlying sklearn/xgb model, expected to be passed from outside or inside wrapper?
    # Wait, X_test_scaled is already scaled. 
    # If we pass the wrapper 'FraudDetector', it expects raw X and handles scaling.
    # If we pass the internal model, it expects scaled X.
    # Let's assume we pass the internal model and scaled data for this low-level eval function.
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Text Report
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    with open(os.path.join(report_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC AUC Score: {auc}")
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'roc_curve.png'))
    plt.close()
    
    return {"auc": auc, "confusion_matrix": cm.tolist()}
