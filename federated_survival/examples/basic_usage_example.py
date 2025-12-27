"""
Basic Usage Example

This example demonstrates the basic usage of the federated survival analysis framework,
including data generation, partitioning, and federated learning.
"""
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


def main():
    """Main function: Demonstrate basic federated learning workflow"""
    
    print("=== Basic Federated Survival Analysis Example ===\n")
    
    # Step 1: Configure data generation
    print("Step 1: Configuring data generation...")
    sim_config = SimulationConfig(
        n_samples=1000,      # Number of samples
        n_features=10,       # Number of features
        random_state=42      # Random seed for reproducibility
    )
    
    # Step 2: Generate simulated data
    print("Step 2: Generating simulated data...")
    generator = DataGenerator(sim_config)
    
    # Generate data with Weibull AFT model
    data_weibull = generator.generate('weibull', c_mean=0.4)
    print(f"  Generated Weibull data: {data_weibull.shape}")
    
    # Step 3: Partition data for federated learning
    print("\nStep 3: Partitioning data for federated learning...")
    splitter = DataSplitter(
        n_clients=3,           # Number of federated learning clients
        split_type='iid',      # Partition type: 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'
        test_size=0.2,         # Proportion of test set
        random_state=42        # Random seed for reproducibility
    )
    
    client_data = splitter.split(data_weibull)
    print(f"  Data split into {len(client_data.clients_set)} clients")
    print(f"  Test set size: {client_data.test_data.shape}")
    
    # Step 4: Configure federated learning
    print("\nStep 4: Configuring federated learning...")
    config = FSAConfig(
        num_clients=3,           # Number of federated learning clients
        n_features=10,           # Number of features
        n_samples=1000,          # Number of samples
        censor_rate=0.2,         # Censoring rate
        model_type='PC-Hazard',  # Survival model type
        local_epochs=2,          # Number of local training epochs
        global_epochs=10,        # Number of global communication rounds
        learning_rate=0.01,      # Learning rate
        batch_size=32,           # Batch size
        random_seed=42,          # Random seed
        client_sample_ratio=1.0, # Ratio of clients selected in each round
        early_stopping=False,    # Disable early stopping for demo
        verbose=True             # Show training progress
    )
    
    print(f"  Model type: {config.model_type}")
    print(f"  Global epochs: {config.global_epochs}")
    print(f"  Local epochs: {config.local_epochs}")
    
    # Step 5: Run federated learning
    print("\nStep 5: Running federated learning...")
    runner = FSARunner(config)
    results = runner.run(client_data, type='raw')
    
    # Step 6: Display results
    print("\n=== Training Results ===")
    print(f"Final Training C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"Final Test C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"Final Training IBS: {results['train_IBS'][-1]:.4f}")
    print(f"Final Test IBS: {results['test_IBS'][-1]:.4f}")
    
    # Step 7: Visualize results
    print("\nStep 7: Visualizing training results...")
    runner.plot_results(results)
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
