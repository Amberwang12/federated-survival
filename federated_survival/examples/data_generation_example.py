"""
Data Generation Example

This example demonstrates different data generation methods available in the framework,
including various simulation types for survival analysis.
"""
from federated_survival.data.generator import DataGenerator, SimulationConfig
import numpy as np


def main():
    """Main function: Demonstrate different data generation methods"""
    
    print("=== Data Generation Methods Example ===\n")
    
    # Configure data generation
    sim_config = SimulationConfig(
        n_samples=100,       # Number of samples
        n_features=10,       # Number of features
        censoring_rate=0.2,  # Rate of censored observations
        random_state=42      # Random seed for reproducibility
    )
    
    generator = DataGenerator(sim_config)
    
    print(f"Configuration:")
    print(f"  Samples: {sim_config.n_samples}")
    print(f"  Features: {sim_config.n_features}")
    print(f"  Censoring rate: {sim_config.censoring_rate}")
    print()
    
    # 1. Accelerated Failure Time (AFT) Models
    print("=" * 60)
    print("1. Accelerated Failure Time (AFT) Models")
    print("=" * 60)
    
    print("\n1.1 Weibull AFT Model")
    data_weibull = generator.generate('weibull')
    print(f"  Data shape: {data_weibull.shape}")
    print(f"  Features: {data_weibull.shape[1] - 2}")
    print(f"  Description: Weibull AFT model with second half of features relevant")
    censoring_rate = 1 - np.mean(data_weibull.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    print("\n1.2 Lognormal AFT Model")
    data_lognormal = generator.generate('lognormal')
    print(f"  Data shape: {data_lognormal.shape}")
    print(f"  Description: Lognormal AFT model with first and last 20% of features relevant")
    censoring_rate = 1 - np.mean(data_lognormal.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    # 2. Proportional Hazards Models
    print("\n" + "=" * 60)
    print("2. Proportional Hazards Models")
    print("=" * 60)
    
    print("\n2.1 Standard Proportional Hazards (SDGM1)")
    data_sdgm1 = generator.generate('SDGM1')
    print(f"  Data shape: {data_sdgm1.shape}")
    print(f"  Description: Standard proportional hazards model")
    censoring_rate = 1 - np.mean(data_sdgm1.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    print("\n2.2 Proportional Hazards with Log-Normal Errors (SDGM4)")
    data_sdgm4 = generator.generate('SDGM4')
    print(f"  Data shape: {data_sdgm4.shape}")
    print(f"  Description: Proportional hazards with log-normal errors")
    censoring_rate = 1 - np.mean(data_sdgm4.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    # 3. Non-Proportional Hazards Models
    print("\n" + "=" * 60)
    print("3. Non-Proportional Hazards Models")
    print("=" * 60)
    
    print("\n3.1 Mild Violations of Proportional Hazards (SDGM2)")
    data_sdgm2 = generator.generate('SDGM2')
    print(f"  Data shape: {data_sdgm2.shape}")
    print(f"  Description: Mild violations with non-linear effects")
    censoring_rate = 1 - np.mean(data_sdgm2.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    print("\n3.2 Strong Violations of Proportional Hazards (SDGM3)")
    data_sdgm3 = generator.generate('SDGM3')
    print(f"  Data shape: {data_sdgm3.shape}")
    print(f"  Description: Strong violations with shape parameter dependency")
    censoring_rate = 1 - np.mean(data_sdgm3.values[:, -1])
    print(f"  Actual censoring rate: {censoring_rate:.2%}")
    
    # Data Structure Explanation
    print("\n" + "=" * 60)
    print("Data Structure")
    print("=" * 60)
    print("\nThe generated data includes:")
    print("  - Features (x1, x2, ..., xp): Generated with AR(1) covariance structure")
    print("  - Time: Observed survival/censoring time (second to last column)")
    print("  - Status: Event indicator (1 = event, 0 = censored, last column)")
    
    # Example data inspection
    print("\nExample data (first 5 rows of Weibull data):")
    print("  Features (first 5 cols) | Time | Status")
    print("-" * 60)
    data_array = data_weibull.values  # Convert to numpy array
    for i in range(min(5, data_array.shape[0])):
        features_str = " ".join([f"{x:6.2f}" for x in data_array[i, :5]])
        time = data_array[i, -2]
        status = int(data_array[i, -1])
        print(f"  {features_str} | {time:6.2f} | {status}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
