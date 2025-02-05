# feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_features(df):
    """
    Preprocesses the features:
      - Scales numerical features.
      - One-hot encodes categorical features.
    Returns the processed feature matrix and the fitted preprocessor.
    """
    # Define numerical and categorical features
    numerical_features = ['cpu_cores', 'cpu_speed', 'memory', 'storage', 'network_speed',
                          'latency', 'throughput', 'error_rate', 'completion_time']
    categorical_features = ['os', 'software_stack']
    
    # Set up transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine them using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit and transform the data
    X = preprocessor.fit_transform(df)
    
    return X, preprocessor
