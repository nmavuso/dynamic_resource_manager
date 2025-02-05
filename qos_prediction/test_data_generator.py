# test_data_generator.py
import numpy as np
import pandas as pd
import argparse

def generate_test_data(num_records=100, random_seed=42):
    """
    Generates synthetic test data for QoS compliance modeling.

    Parameters:
        num_records (int): Number of data records to generate.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with synthetic server configuration and performance data.
    """
    np.random.seed(random_seed)
    
    # Simulate server configuration features
    cpu_cores = np.random.choice([4, 8, 16], size=num_records)
    cpu_speed = np.random.choice([2.5, 3.0, 3.5], size=num_records)  # in GHz
    memory = np.random.choice([16, 32, 64], size=num_records)         # in GB
    # Representing storage in GB: for example, 1024 (1TB HDD), 500 (SSD), etc.
    storage = np.random.choice([1024, 500, 1024], size=num_records)
    network_speed = np.random.choice([1, 10, 40], size=num_records)     # in Gbps

    # Categorical features: Operating System and Software Stack
    os_options = ['Linux', 'Windows']
    software_options = ['Apache_MySQL', 'Nginx_PostgreSQL']
    os_feature = np.random.choice(os_options, size=num_records)
    software_stack = np.random.choice(software_options, size=num_records)
    
    # Generate QoS performance metrics using normal distributions
    latency = np.random.normal(loc=150, scale=50, size=num_records)            # in ms
    throughput = np.random.normal(loc=120, scale=30, size=num_records)           # requests per second
    error_rate = np.abs(np.random.normal(loc=0.03, scale=0.02, size=num_records))# as a fraction
    completion_time = np.random.normal(loc=900, scale=200, size=num_records)     # in ms
    
    # Ensure no negative values
    latency = np.clip(latency, 0, None)
    throughput = np.clip(throughput, 0, None)
    completion_time = np.clip(completion_time, 0, None)
    
    # Define QoS thresholds
    qos_thresholds = {
        'latency': 200,         # Maximum acceptable latency in ms
        'throughput': 100,      # Minimum acceptable throughput (req/sec)
        'error_rate': 0.05,     # Maximum acceptable error rate (5%)
        'completion_time': 1000 # Maximum acceptable job completion time in ms
    }
    
    # Label the data: 1 if all QoS criteria are met, else 0.
    qos_compliant = (
        (latency <= qos_thresholds['latency']) &
        (throughput >= qos_thresholds['throughput']) &
        (error_rate <= qos_thresholds['error_rate']) &
        (completion_time <= qos_thresholds['completion_time'])
    ).astype(int)
    
    # Build the DataFrame
    data = {
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
        'completion_time': completion_time,
        'QoS_compliant': qos_compliant
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data for QoS compliance modeling.")
    parser.add_argument("--num", type=int, default=100, help="Number of test records to generate (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation (default: 42)")
    parser.add_argument("--output", type=str, default="test_data.csv", help="Output CSV file path (default: test_data.csv)")
    args = parser.parse_args()
    
    # Generate and save the test data
    df = generate_test_data(num_records=args.num, random_seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {args.num} test data records and saved to {args.output}")

if __name__ == '__main__':
    main()
