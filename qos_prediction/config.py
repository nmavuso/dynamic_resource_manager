# config.py
import os

# QoS thresholds (example values)
QOS_THRESHOLDS = {
    'latency': 200,         # in milliseconds: max acceptable latency
    'throughput': 100,      # in requests per second: min acceptable throughput
    'error_rate': 0.05,     # max acceptable error rate (5%)
    'completion_time': 1000 # in milliseconds: max acceptable job completion time
}

# Dataset configuration
DATASET_SIZE = 1000  # Number of synthetic records to generate
TEST_SIZE = 0.2      # Fraction of data to use for testing
RANDOM_SEED = 42     # Seed for reproducibility

# Model configuration parameters (for Random Forest)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': RANDOM_SEED
}

# List of feature names (both configuration and performance metrics)
FEATURE_COLUMNS = [
    'cpu_cores', 'cpu_speed', 'memory', 'storage', 'network_speed',
    'os', 'software_stack', 'latency', 'throughput', 'error_rate', 'completion_time'
]
