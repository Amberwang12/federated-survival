"""
Example: Perturbation of Historical Client Updates (No End-to-End DP Guarantee)

This example only demonstrates how to invoke three legacy noise/selection tools.
The current training path has no per-sample clipping, privacy amplification,
or cross-round accountant, so the epsilon/delta parameters must NOT be
reported as record-level DP guarantees:
1. Gaussian Mechanism - for deep learning gradient protection
2. Laplace Mechanism - for counting queries
3. Exponential Mechanism - for model selection
"""
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
import torch
import numpy as np


def generate_data(n_samples=1000, n_features=20, num_clients=5):
    """Generate federated learning data.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        num_clients: Number of clients

    Returns:
        Split federated learning data
    """
    sim_config = SimulationConfig(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull', c_mean=0.4)
    
    splitter = DataSplitter(
        n_clients=num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=42
    )
    return splitter.split(raw_data)


def demonstrate_gaussian_mechanism():
    """Demonstrate the Gaussian Mechanism.

    Demonstrates the Gaussian noise tool; the current federated training
    path does not provide (epsilon, delta)-DP guarantees based on it.
    Suitable for deep learning gradient protection by adding Gaussian
    noise to model parameters.
    """
    print("\n" + "="*60)
    print("1. Gaussian Mechanism")
    print("="*60)
    print("Properties: (epsilon, delta)-differential privacy, Gaussian noise")
    print("Use case: Deep learning gradient protection\n")

    # Create configuration
    config = FSAConfig(
        n_samples=1000,
        n_features=20,
        num_clients=5,
        global_epochs=15,
        verbose=False,
        use_differential_privacy=True,
        dp_mechanism='gaussian',      # Gaussian mechanism
        dp_epsilon=1.0,
        dp_delta=1e-5,                # Gaussian mechanism requires delta
        dp_noise_multiplier=1.0,      # Gaussian noise multiplier
        dp_clip_norm=1.0,
    )

    # Generate data
    print("Generating data...")
    data = generate_data(config.n_samples, config.n_features, config.num_clients)

    # Run training
    print("Starting training...")
    runner = FSARunner(config)
    results = runner.run(data, type='raw')

    # Show privacy information
    privacy_info = runner.get_privacy_info()
    print("\nPrivacy protection info:")
    print(f"  Mechanism: {privacy_info['mechanism']}")
    print(f"  Privacy budget (epsilon): {privacy_info['epsilon']}")
    print(f"  Failure probability (delta): {privacy_info['delta']}")
    print(f"  Noise multiplier: {privacy_info['noise_multiplier']}")
    print(f"  Gradient clipping norm: {privacy_info['clip_norm']}")

    # Show results
    print("\nTraining results:")
    print(f"  Final train C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  Final test C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  Final train IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  Final test IBS: {results['test_IBS'][-1]:.4f}")
    
    return results


def demonstrate_laplace_mechanism():
    """Demonstrate the Laplace Mechanism.

    Demonstrates the Laplace noise tool; the current federated training
    path does not provide pure epsilon-DP guarantees based on it.
    Suitable for counting and sum queries; the noise follows a
    Laplace distribution.
    """
    print("\n" + "="*60)
    print("2. Laplace Mechanism")
    print("="*60)
    print("Properties: epsilon-differential privacy, Laplace-distributed noise")
    print("Use case: Counting queries, sum queries\n")

    # Create configuration
    config = FSAConfig(
        n_samples=1000,
        n_features=20,
        num_clients=5,
        global_epochs=15,
        verbose=False,
        use_differential_privacy=True,
        dp_mechanism='laplace',       # Laplace mechanism
        dp_epsilon=1.0,
        # Note: the Laplace mechanism does not require delta or noise_multiplier
        dp_clip_norm=1.0,
    )

    # Generate data
    print("Generating data...")
    data = generate_data(config.n_samples, config.n_features, config.num_clients)

    # Run training
    print("Starting training...")
    runner = FSARunner(config)
    results = runner.run(data, type='raw')

    # Show privacy information
    privacy_info = runner.get_privacy_info()
    print("\nPrivacy protection info:")
    print(f"  Mechanism: {privacy_info['mechanism']}")
    print(f"  Privacy budget (epsilon): {privacy_info['epsilon']}")
    print(f"  Gradient clipping norm: {privacy_info['clip_norm']}")
    print(f"  Note: the Laplace mechanism provides pure epsilon-DP; no delta parameter is needed")

    # Show results
    print("\nTraining results:")
    print(f"  Final train C-index: {results['train_Cindex'][-1]:.4f}")
    print(f"  Final test C-index: {results['test_Cindex'][-1]:.4f}")
    print(f"  Final train IBS: {results['train_IBS'][-1]:.4f}")
    print(f"  Final test IBS: {results['test_IBS'][-1]:.4f}")
    
    return results


def demonstrate_exponential_mechanism():
    """Demonstrate the Exponential Mechanism.

    Demonstrates discrete candidate selection; its DP guarantees can only
    be claimed if the sensitivity of the quality function is proven separately.
    Commonly used for discrete optimization problems such as model selection
    and hyperparameter selection.
    """
    print("\n" + "="*60)
    print("3. Exponential Mechanism")
    print("="*60)
    print("Properties: epsilon-differential privacy, probabilistic sampling")
    print("Use case: Model selection, hyperparameter selection, discrete optimization\n")

    # Create configuration for initializing the differential privacy tool
    config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='exponential',
        dp_epsilon=2.0,
        dp_sensitivity=1.0,
    )

    # Create the differential privacy tool
    dp_tool = DifferentialPrivacy(config)

    # Define candidate model names and quality scores
    model_names = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
    quality_scores = torch.tensor([0.75, 0.82, 0.68, 0.79, 0.85])  # Model quality scores

    # Create candidate tensor (a simple index tensor is used here)
    candidates = torch.arange(len(model_names))

    print("Candidate models and quality scores:")
    for i, (model, score) in enumerate(zip(model_names, quality_scores)):
        print(f"  {model}: {score:.4f}")
    print()

    # Select a model via the exponential mechanism (sample repeatedly to observe the probability distribution)
    print("Selecting a model via the exponential mechanism (100 samples)...")
    n_trials = 100
    selection_counts = {model: 0 for model in model_names}

    for _ in range(n_trials):
        selected_idx = dp_tool.exponential_mechanism(
            candidates=candidates,
            quality_scores=quality_scores
        )
        selection_counts[model_names[selected_idx]] += 1

    # Show selection statistics
    print("\nSelection statistics (based on privacy budget epsilon=2.0):")
    for model in model_names:
        percentage = (selection_counts[model] / n_trials) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {model}: {selection_counts[model]:3d} times ({percentage:5.1f}%) {bar}")

    # Theoretical analysis
    print("\nTheoretical analysis:")
    print("  The exponential mechanism's selection probability grows exponentially with the quality score")
    print(f"  The highest-scoring model (Model_E: {quality_scores[4]:.4f}) has the highest selection probability")
    print(f"  Actual selection count: {selection_counts['Model_E']} times")
    print("  Note: this example does not establish an end-to-end privacy guarantee")
    
    return selection_counts


def main():
    """Main function: demonstrate three differential privacy mechanisms."""
    print("\n" + "#"*60)
    print("# Differential Privacy in Federated Learning for Survival Analysis - Three Mechanisms Demo")
    print("#"*60)
    print("\nThis example demonstrates three differential privacy mechanisms in federated learning:")
    print("  1. Gaussian Mechanism")
    print("  2. Laplace Mechanism")
    print("  3. Exponential Mechanism")
    print("\nThese tools serve different experimental purposes, but the current training")
    print("pipeline has no end-to-end privacy accounting.")

    # Demonstrate the three mechanisms
    results_gaussian = demonstrate_gaussian_mechanism()
    results_laplace = demonstrate_laplace_mechanism()
    selection_counts = demonstrate_exponential_mechanism()

    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print("\nMechanism properties comparison:")
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")
    print("| {:^16} | {:^16} | {:^18} |".format("Tool", "Guarantee here", "Experimental use"))
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")
    print("| {:^16} | {:^16} | {:^18} |".format("Gaussian", "(epsilon, delta)-DP", "DL gradients"))
    print("| {:^16} | {:^16} | {:^18} |".format("Laplace", "epsilon-DP", "Count/sum queries"))
    print("| {:^16} | {:^16} | {:^18} |".format("Exponential", "epsilon-DP", "Model selection"))
    print("+" + "-"*18 + "+" + "-"*18 + "+" + "-"*20 + "+")

    print("\nPerformance comparison (based on this run):")
    print(f"  Gaussian mechanism - test C-index: {results_gaussian['test_Cindex'][-1]:.4f}")
    print(f"  Laplace mechanism - test C-index: {results_laplace['test_Cindex'][-1]:.4f}")
    print(f"  Exponential mechanism - model selection: {selection_counts['Model_E']} times")

    print("\nUsage recommendations:")
    print("  • Deep learning model training → choose the Gaussian Mechanism")
    print("  • Simple statistical queries → choose the Laplace Mechanism")
    print("  • Model or parameter selection → choose the Exponential Mechanism")

    print("\n" + "#"*60)
    print("Example run completed!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
