# evaluator.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

def evaluate_model(clf, X_test, y_test):
    try:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logging.info("Model evaluation metrics: %s", metrics)
        return metrics
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise
