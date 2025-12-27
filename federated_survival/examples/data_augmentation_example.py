"""
Data Augmentation Example - Shortened version
"""
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


def main():
    """Main function: Demonstrate data augmentation"""
    
    print("=== Data Augmentation Example ===\n")
    
    # Generate data
    sim_config = SimulationConfig(n_samples=500, n_features=10, random_state=42)
    generator = DataGenerator(sim_config)
    data = generator.generate('weibull', c_mean=0.4)
    # data = generator.generate('lognormal', c_mean=0.4)
    # data = generator.generate('SDGM1', c_mean=0.4)
    # data = generator.generate('SDGM2', u_max=4)
    # data = generator.generate('SDGM3', u_max=7)
    # data = generator.generate('SDGM4', c_step=0.4)
    
    splitter = DataSplitter(n_clients=3, split_type='iid', test_size=0.2, random_state=42)
    client_data = splitter.split(data)
    
    # Configure with augmentation parameters
    config = FSAConfig(
        num_clients=3, n_features=10, model_type='PC-Hazard',
        global_epochs=10, latent_num=10, hidden_num=30, alpha=1.0, beta=1.0, k=0.5
    )
    
    runner = FSARunner(config)
    
    # Test MVAEC
    print("\nRunning with MVAEC augmentation...")
    results_mvaec = runner.run(client_data, type='raw_aug', aug_method='MVAEC')
    print(f"Test C-index: {results_mvaec['test_Cindex'][-1]:.4f}")
    
    print("\n=== Example completed! ===")


if __name__ == "__main__":
    main()
