"""
Example comparing the effect of perturbing client updates
(no end-to-end differential privacy guarantee).

This example compares performance with and without client update
perturbation. The current implementation has no per-example clipping or
cross-round accountant, so the epsilon/delta parameters must not be
interpreted as record-level DP guarantees.
"""
import matplotlib.pyplot as plt
import numpy as np
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter

def run_experiment(config, data, experiment_name):
    """Run an experiment and return the results."""
    print(f"\n=== {experiment_name} ===")
    runner = FSARunner(config)
    
    # Get privacy information
    privacy_info = runner.get_privacy_info()
    if privacy_info["privacy_protection"]:
        print("Experimental client update perturbation enabled (not an end-to-end DP guarantee)")
        print(f"Legacy epsilon parameter: {privacy_info['epsilon']}")
        print(f"Noise scale: {privacy_info['noise_scale']:.6f}")
    else:
        print("Client update perturbation is not enabled")
    
    # Run training
    results = runner.run(data, type='raw')
    
    print(f"Final performance:")
    print(f"  Training C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  Test C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  Training IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  Test IBS: {results['test_IBS'][-1]:.4f}")
    
    return results

def plot_comparison(results_no_dp, results_with_dp):
    """Plot the comparison results."""
    # Use a more modern style, avoiding the deprecated seaborn style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('default')
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Close all existing figure windows
    plt.close('all')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = list(range(1, len(results_no_dp['train_Cindex']) + 1))
    
    # C-index Comparison
    ax1.plot(epochs, results_no_dp['train_Cindex'], 'b-', label='No DP (Training)', linewidth=2)
    ax1.plot(epochs, results_no_dp['test_Cindex'], 'b--', label='No DP (Testing)', linewidth=2)
    ax1.plot(epochs, results_with_dp['train_Cindex'], 'r-', label='With DP (Training)', linewidth=2)
    ax1.plot(epochs, results_with_dp['test_Cindex'], 'r--', label='With DP (Testing)', linewidth=2)
    ax1.set_title('C-index Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('C-index')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IBS Comparison
    ax2.plot(epochs, results_no_dp['train_IBS'], 'b-', label='No DP (Training)', linewidth=2)
    ax2.plot(epochs, results_no_dp['test_IBS'], 'b--', label='No DP (Testing)', linewidth=2)
    ax2.plot(epochs, results_with_dp['train_IBS'], 'r-', label='With DP (Training)', linewidth=2)
    ax2.plot(epochs, results_with_dp['test_IBS'], 'r--', label='With DP (Testing)', linewidth=2)
    ax2.set_title('IBS Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IBS')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance Difference
    cindex_diff = np.array(results_with_dp['test_Cindex']) - np.array(results_no_dp['test_Cindex'])
    ibs_diff = np.array(results_with_dp['test_IBS']) - np.array(results_no_dp['test_IBS'])
    
    ax3.plot(epochs, cindex_diff, 'g-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('C-index Difference (With DP - No DP)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('C-index Difference')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(epochs, ibs_diff, 'orange', linewidth=2, marker='s', markersize=4)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('IBS Difference (With DP - No DP)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IBS Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function: compare results with and without differential privacy."""
    print("=== Differential Privacy Effect Comparison Experiment ===\n")

    # Base configuration
    base_config = {
        'n_samples': 1000,
        'n_features': 20,
        'num_clients': 5,
        'global_epochs': 30,
        'verbose': True,
    }
    
    # Generate data
    print("Generating simulated data...")
    temp_config = FSAConfig(**base_config)
    sim_config = SimulationConfig(
        n_samples=temp_config.n_samples,
        n_features=temp_config.n_features,
        random_state=temp_config.random_seed
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull', c_mean=0.4)

    # Split data into federated learning format
    splitter = DataSplitter(
        n_clients=temp_config.num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=temp_config.random_seed
    )
    data = splitter.split(raw_data)
    print(f"Data generation complete, number of clients: {len(data.clients_set)}")
    
    # Experiment 1: without differential privacy
    config_no_dp = FSAConfig(**base_config, use_differential_privacy=False)
    results_no_dp = run_experiment(config_no_dp, data, "Without differential privacy")

    # Experiment 2: with differential privacy
    config_with_dp = FSAConfig(
        **base_config,
        use_differential_privacy=True,
        dp_epsilon=0.1,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0,
    )
    results_with_dp = run_experiment(config_with_dp, data, "With differential privacy")

    # Plot the comparison results
    print("\nPlotting comparison results...")
    plot_comparison(results_no_dp, results_with_dp)
    
    # Print final comparison
    print("\n=== Final Performance Comparison ===")
    print(f"{'Metric':<15} {'No DP':<12} {'With DP':<12} {'Difference':<12}")
    print("-" * 55)
    
    train_cindex_diff = results_with_dp['train_Cindex'][-1] - results_no_dp['train_Cindex'][-1]
    test_cindex_diff = results_with_dp['test_Cindex'][-1] - results_no_dp['test_Cindex'][-1]
    train_ibs_diff = results_with_dp['train_IBS'][-1] - results_no_dp['train_IBS'][-1]
    test_ibs_diff = results_with_dp['test_IBS'][-1] - results_no_dp['test_IBS'][-1]
    
    print(f"{'Training C-index':<15} {results_no_dp['train_Cindex'][-1]:<12.4f} {results_with_dp['train_Cindex'][-1]:<12.4f} {train_cindex_diff:+.4f}")
    print(f"{'Test C-index':<15} {results_no_dp['test_Cindex'][-1]:<12.4f} {results_with_dp['test_Cindex'][-1]:<12.4f} {test_cindex_diff:+.4f}")
    print(f"{'Training IBS':<15} {results_no_dp['train_IBS'][-1]:<12.4f} {results_with_dp['train_IBS'][-1]:<12.4f} {train_ibs_diff:+.4f}")
    print(f"{'Test IBS':<15} {results_no_dp['test_IBS'][-1]:<12.4f} {results_with_dp['test_IBS'][-1]:<12.4f} {test_ibs_diff:+.4f}")

    print("\nExperiment completed!")

if __name__ == "__main__":
    main()
