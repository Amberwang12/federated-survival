"""
Data Partitioning Example

This example demonstrates different data partitioning methods to simulate
various federated learning scenarios (IID, Non-IID, Time-Non-IID, Dirichlet).
"""
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
import numpy as np


def analyze_partition(client_data, partition_name):
    """Analyze and display partition statistics"""
    print(f"\n{partition_name} Partition Analysis:")
    print("-" * 60)
    
    # Client statistics
    print(f"Number of clients: {len(client_data.clients_set)}")
    print(f"Test set size: {client_data.test_data.shape[0]}")
    
    # Per-client statistics
    print("\nPer-client statistics:")
    print(f"{'Client':<10} {'Samples':<10} {'Censoring Rate':<20} {'Avg Time':<10}")
    print("-" * 60)
    
    for client_id, (X, y) in client_data.clients_set.items():
        n_samples = X.shape[0]
        censoring_rate = 1 - np.mean(y[:, 1])  # 1 - event rate
        avg_time = np.mean(y[:, 0])
        print(f"{client_id:<10} {n_samples:<10} {censoring_rate:<20.2%} {avg_time:<10.2f}")


def main():
    """Main function: Demonstrate different data partitioning methods"""
    
    print("=== Data Partitioning Methods Example ===\n")
    
    # Generate base dataset
    print("Generating base dataset...")
    sim_config = SimulationConfig(
        n_samples=1000,
        n_features=10,
        random_state=42
    )
    
    generator = DataGenerator(sim_config)
    data = generator.generate('weibull', c_mean=0.4)
    print(f"Base dataset shape: {data.shape}")
    print(f"Total samples: {data.shape[0]}")
    print()
    
    # 1. IID (Independent and Identically Distributed)
    print("=" * 60)
    print("1. IID Partition")
    print("=" * 60)
    print("Description: Ensures each client has the same censoring rate")
    print("Use case: Ideal federated learning scenarios")
    
    splitter_iid = DataSplitter(
        n_clients=5,
        split_type='iid',
        test_size=0.2,
        random_state=42
    )
    client_data_iid = splitter_iid.split(data)
    analyze_partition(client_data_iid, "IID")
    
    # 2. Non-IID
    print("\n" + "=" * 60)
    print("2. Non-IID Partition")
    print("=" * 60)
    print("Description: Random splits without maintaining censoring rate balance")
    print("Use case: Testing model robustness to distribution shifts")
    
    splitter_non_iid = DataSplitter(
        n_clients=5,
        split_type='non-iid',
        test_size=0.2,
        random_state=42
    )
    client_data_non_iid = splitter_non_iid.split(data)
    analyze_partition(client_data_non_iid, "Non-IID")
    
    # 3. Time-Non-IID
    print("\n" + "=" * 60)
    print("3. Time-Non-IID Partition")
    print("=" * 60)
    print("Description: Splits data based on survival time ranges")
    print("Use case: Simulating temporal distribution shifts")
    
    splitter_time_non_iid = DataSplitter(
        n_clients=5,
        split_type='time-non-iid',
        test_size=0.2,
        random_state=42
    )
    client_data_time_non_iid = splitter_time_non_iid.split(data)
    analyze_partition(client_data_time_non_iid, "Time-Non-IID")
    
    # 4. Dirichlet
    print("\n" + "=" * 60)
    print("4. Dirichlet Partition (Experimental)")
    print("=" * 60)
    print("Description: Uses Dirichlet distribution for non-IID splits")
    print("Use case: Creating complex non-IID scenarios")
    
    splitter_dirichlet = DataSplitter(
        n_clients=5,
        split_type='Dirichlet',
        alpha=0.5,  # Controls degree of non-IID
        test_size=0.2,
        random_state=42
    )
    client_data_dirichlet = splitter_dirichlet.split(data)
    analyze_partition(client_data_dirichlet, "Dirichlet")
    
    # Data Structure Explanation
    print("\n" + "=" * 60)
    print("Data Structure")
    print("=" * 60)
    print("\nThe partitioned data structure:")
    print("  - clients_set: Dictionary of client data")
    print("    * Key: client_id (e.g., 'client0', 'client1', ...)")
    print("    * Value: Tuple of (features, labels)")
    print("      - features (X): numpy array of shape (n_samples, n_features)")
    print("      - labels (y): numpy array of shape (n_samples, 2)")
    print("        - First column: survival/censoring time")
    print("        - Second column: event indicator (1 = event, 0 = censored)")
    print("  - test_data: Test set features")
    print("  - test_label: Test set labels (time and status)")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
