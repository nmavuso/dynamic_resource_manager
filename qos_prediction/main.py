# main.py

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer and have PyQt5 installed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
from sklearn.model_selection import train_test_split
from data_loader import load_data
from feature_engineering import preprocess_features
from model_trainer import train_random_forest, train_decision_tree
from evaluator import evaluate_model
from config import TEST_SIZE, RANDOM_SEED
from utils import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="QoS Compliance Prediction Experiment")
    parser.add_argument('--data', type=str, default=None,
                        help="Path to the CSV dataset. If not provided, synthetic data is generated.")
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'dt'],
                        help="Model type: 'rf' for Random Forest, 'dt' for Decision Tree.")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    logging.info("Starting QoS Compliance Prediction Experiment")
    
    # Load data (from file if provided; otherwise, generate synthetic data)
    df = load_data(args.data)
    logging.info("Data loaded successfully with shape %s", df.shape)
    
    # Ensure the target column exists
    target_column = 'QoS_compliant'
    if target_column not in df.columns:
        logging.error("Target column '%s' not found in the data", target_column)
        return
    
    # Split data into features and target
    X_raw = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Preprocess features
    X_processed, preprocessor = preprocess_features(X_raw)
    logging.info("Feature preprocessing completed.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    logging.info("Data split into training and testing sets.")
    
    # Train the selected model
    if args.model == 'rf':
        clf = train_random_forest(X_train, y_train)
    else:
        clf = train_decision_tree(X_train, y_train)
    
    # Evaluate the trained model
    metrics = evaluate_model(clf, X_test, y_test)
    logging.info("Experiment completed. Evaluation metrics:")
    for metric, value in metrics.items():
        logging.info("%s: %.4f", metric, value)

    # Add these lines after model evaluation in main.py
    try:
    # Get predicted probabilities for the positive class
         y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Compute the ROC curve and AUC score
         fpr, tpr, _ = roc_curve(y_test, y_proba)
         roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
         plt.figure(figsize=(8, 6))
         plt.plot(fpr, tpr, lw=2, color='darkorange',
             label=f'ROC curve (area = {roc_auc:.2f})')
         plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle='--',
             label='Random guess')
         plt.xlabel('False Positive Rate')
         plt.ylabel('True Positive Rate')
         plt.title('Receiver Operating Characteristic (ROC)')
         plt.legend(loc='lower right')
         plt.show()
    except AttributeError:
         print("The chosen model does not support probability estimates (predict_proba).")


if __name__ == '__main__':
    main()
