# data_loader.py
import pandas as pd
import numpy as np
from config import DATASET_SIZE, RANDOM_SEED, QOS_THRESHOLDS

def generate_synthetic_data(size=DATASET_SIZE, seed=RANDOM_SEED):
    np.random.seed(seed)
    
    # Simulate server configuration features
    cpu_cores = np.random.choice([4, 8, 16], size=size)
    cpu_speed = np.random.choice([2.5, 3.0, 3.5], size=size)  # in GHz
    memory = np.random.choice([16, 32, 64], size=size)         # in GB
    # For storage, we simulate values in GB (e.g., 1TB HDD, 500GB SSD, etc.)
    storage = np.random.choice([1024, 500, 1024], size=size)
    network_speed = np.random.choice([1, 10, 40], size=size)     # in Gbps
    
    # Categorical features: OS and software stack
    os_options = ['Linux', 'Windows']
    software_options = ['Apache_MySQL', 'Nginx_PostgreSQL']
    os_feature = np.random.choice(os_options, size=size)
    software_stack = np.random.choice(software_options, size=size)
    
    # Simulate QoS performance metrics using normal distributions (with noise)
    latency = np.random.normal(loc=150, scale=50, size=size)            # ms
    throughput = np.random.normal(loc=120, scale=30, size=size)           # requests per second
    error_rate = np.abs(np.random.normal(loc=0.03, scale=0.02, size=size))# fraction (ensure positive)
    completion_time = np.random.normal(loc=900, scale=200, size=size)     # ms
    
    # Avoid negative values
    latency = np.clip(latency, 0, None)
    throughput = np.clip(throughput, 0, None)
    completion_time = np.clip(completion_time, 0, None)
    
    # Build the DataFrame
    df = pd.DataFrame({
        'cpu_cores': cpu_cores,
        'cpu_speed': cpu_speed,
        'memory': memory,
        'storage': storage,
        'network_speed': network_speed,
        'os': os_feature,
        'software_stack': software_stack,
        'latency': latency,
        'throughput': throughput,
        'error_rate': error_rate,
        'completion_time': completion_time
    })
    
    # Label data: A job is QoS compliant if all metrics meet their respective thresholds.
    df['QoS_compliant'] = (
        (df['latency'] <= QOS_THRESHOLDS['latency']) &
        (df['throughput'] >= QOS_THRESHOLDS['throughput']) &
        (df['error_rate'] <= QOS_THRESHOLDS['error_rate']) &
        (df['completion_time'] <= QOS_THRESHOLDS['completion_time'])
    ).astype(int)
    
    return df

def load_data(filepath=None):
    """
    Loads data from a CSV file if a filepath is provided.
    Otherwise, generates synthetic data.
    """
    if filepath:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise Exception(f"Error reading file {filepath}: {e}")
    else:
        df = generate_synthetic_data()
    return df
