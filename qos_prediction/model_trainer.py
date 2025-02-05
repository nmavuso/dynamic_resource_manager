# model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from config import MODEL_PARAMS
import logging

def train_random_forest(X_train, y_train):
    try:
        clf = RandomForestClassifier(**MODEL_PARAMS)
        clf.fit(X_train, y_train)
        logging.info("Random Forest training completed successfully.")
        return clf
    except Exception as e:
        logging.error("Error training Random Forest: %s", e)
        raise

def train_decision_tree(X_train, y_train):
    try:
        clf = DecisionTreeClassifier(random_state=MODEL_PARAMS.get('random_state', 42))
        clf.fit(X_train, y_train)
        logging.info("Decision Tree training completed successfully.")
        return clf
    except Exception as e:
        logging.error("Error training Decision Tree: %s", e)
        raise
