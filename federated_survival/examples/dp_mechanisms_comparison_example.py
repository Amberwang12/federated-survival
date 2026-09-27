"""
Legacy Perturbation Utilities Comparison (not an end-to-end DP guarantee)

This example demonstrates legacy noise/selection utilities. The federated
training path has no per-example clipping or multi-round accountant, so the
epsilon/delta inputs must not be reported as a record-level DP guarantee:
1. Gaussian Mechanism (高斯机制) - for (ε,δ)-differential privacy
2. Laplace Mechanism (拉普拉斯机制) - for ε-differential privacy  
3. Exponential Mechanism (指数机制) - for non-numeric outputs

Author: Federated Survival Analysis Team
Date: 2025-10-19
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.core.config import FSAConfig


def demonstrate_noise_mechanisms():
    """
    Demonstrate the three differential privacy mechanisms with visualization
    """
    print("=" * 80)
    print("Differential Privacy Mechanisms Comparison")
    print("=" * 80)
    
    # Create configuration
    config = FSAConfig(
        n_features=10,
        num_clients=5,
        global_epochs=10,
        local_epochs=5,
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0
    )
    
    # Initialize DP tool
    dp_tool = DifferentialPrivacy(config)
    
    # Original data (simulating model parameters or query results)
    original_data = torch.randn(1000) * 10
    
    print("\n1. Gaussian Mechanism (高斯机制)")
    print("-" * 80)
    print(f"Privacy guarantee: (ε={config.dp_epsilon}, δ={config.dp_delta})-DP")
    print(f"Noise distribution: Normal(0, σ²)")
    print(f"Use case: Deep learning, continuous outputs")
    
    # Apply Gaussian mechanism
    gaussian_noisy = dp_tool.add_gaussian_noise(original_data.clone())
    gaussian_error = torch.mean(torch.abs(gaussian_noisy - original_data)).item()
    print(f"Mean absolute error: {gaussian_error:.4f}")
    
    print("\n2. Laplace Mechanism (拉普拉斯机制)")
    print("-" * 80)
    print(f"Privacy guarantee: ε={config.dp_epsilon}-DP")
    print(f"Noise distribution: Laplace(0, b), b = Δf/ε")
    print(f"Use case: Counting queries, sum queries")
    
    # Apply Laplace mechanism
    laplace_noisy = dp_tool.add_laplace_noise(original_data.clone())
    laplace_error = torch.mean(torch.abs(laplace_noisy - original_data)).item()
    print(f"Mean absolute error: {laplace_error:.4f}")
    
    print("\n3. Exponential Mechanism (指数机制)")
    print("-" * 80)
    print(f"Privacy guarantee: ε={config.dp_epsilon}-DP")
    print(f"Selection method: Probability sampling based on quality scores")
    print(f"Use case: Model selection, hyperparameter tuning")
    
    # Apply Exponential mechanism (select best among candidates)
    # Simulate 10 candidate configurations with quality scores
    candidates = torch.randn(10, 100)
    quality_scores = torch.tensor([0.75, 0.80, 0.85, 0.90, 0.88, 0.82, 0.78, 0.76, 0.84, 0.86])
    
    # Without privacy
    best_idx_no_privacy = torch.argmax(quality_scores).item()
    print(f"Best candidate without privacy: Index {best_idx_no_privacy} (score: {quality_scores[best_idx_no_privacy]:.4f})")
    
    # With exponential mechanism
    selected_idx = dp_tool.exponential_mechanism(candidates, quality_scores, epsilon=config.dp_epsilon)
    print(f"Selected candidate with DP: Index {selected_idx} (score: {quality_scores[selected_idx]:.4f})")
    
    # Visualize noise distributions
    visualize_noise_distributions(dp_tool, original_data)
    
    # Compare privacy-utility tradeoff
    compare_privacy_utility(dp_tool, original_data)


def visualize_noise_distributions(dp_tool: DifferentialPrivacy, data: torch.Tensor):
    """
    Visualize the noise distributions of different mechanisms
    """
    print("\n" + "=" * 80)
    print("Visualizing Noise Distributions")
    print("=" * 80)
    
    # Generate samples
    n_samples = 10000
    test_tensor = torch.zeros(n_samples)
    
    # Gaussian noise
    gaussian_samples = dp_tool.add_gaussian_noise(test_tensor.clone())
    
    # Laplace noise
    laplace_samples = dp_tool.add_laplace_noise(test_tensor.clone())
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Gaussian distribution
    plt.subplot(1, 3, 1)
    plt.hist(gaussian_samples.numpy(), bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Gaussian Mechanism\n高斯机制', fontsize=12, fontweight='bold')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Laplace distribution
    plt.subplot(1, 3, 2)
    plt.hist(laplace_samples.numpy(), bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title('Laplace Mechanism\n拉普拉斯机制', fontsize=12, fontweight='bold')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Exponential mechanism (selection probability)
    plt.subplot(1, 3, 3)
    quality_scores = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65])
    candidates = torch.randn(len(quality_scores), 10)
    
    # Run exponential mechanism multiple times to estimate probabilities
    n_trials = 1000
    selection_counts = np.zeros(len(quality_scores))
    for _ in range(n_trials):
        idx = dp_tool.exponential_mechanism(candidates, quality_scores)
        selection_counts[idx] += 1
    
    selection_probs = selection_counts / n_trials
    plt.bar(range(len(quality_scores)), selection_probs, alpha=0.7, color='orange', edgecolor='black')
    plt.plot(range(len(quality_scores)), quality_scores.numpy() / quality_scores.sum().item(), 
             'r--', linewidth=2, label='Quality Scores (normalized)')
    plt.title('Exponential Mechanism\n指数机制', fontsize=12, fontweight='bold')
    plt.xlabel('Candidate Index')
    plt.ylabel('Selection Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dp_mechanisms_noise_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: dp_mechanisms_noise_distributions.png")
    plt.show()


def compare_privacy_utility(dp_tool: DifferentialPrivacy, data: torch.Tensor):
    """
    Compare privacy-utility tradeoff for different epsilon values
    """
    print("\n" + "=" * 80)
    print("Privacy-Utility Tradeoff Comparison")
    print("=" * 80)
    
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    gaussian_errors = []
    laplace_errors = []
    
    for epsilon in epsilon_values:
        # Gaussian mechanism
        gaussian_noisy = dp_tool.add_gaussian_noise(data.clone(), sensitivity=1.0)
        gaussian_error = torch.mean(torch.abs(gaussian_noisy - data)).item()
        gaussian_errors.append(gaussian_error)
        
        # Laplace mechanism
        laplace_noisy = dp_tool.add_laplace_noise(data.clone(), sensitivity=1.0, epsilon=epsilon)
        laplace_error = torch.mean(torch.abs(laplace_noisy - data)).item()
        laplace_errors.append(laplace_error)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, gaussian_errors, 'o-', linewidth=2, markersize=8, 
             label='Gaussian Mechanism', color='blue')
    plt.plot(epsilon_values, laplace_errors, 's-', linewidth=2, markersize=8,
             label='Laplace Mechanism', color='green')
    
    plt.xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    plt.title('Privacy-Utility Tradeoff\n隐私-效用权衡', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add annotations
    plt.text(0.15, max(gaussian_errors) * 0.8, 
             'Higher Privacy\n(More Noise)', 
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(8, min(gaussian_errors) * 1.5,
             'Lower Privacy\n(Less Noise)',
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dp_mechanisms_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: dp_mechanisms_privacy_utility_tradeoff.png")
    plt.show()
    
    # Print summary table
    print("\nPrivacy-Utility Tradeoff Summary:")
    print("-" * 80)
    print(f"{'Epsilon':<10} {'Gaussian Error':<20} {'Laplace Error':<20}")
    print("-" * 80)
    for eps, g_err, l_err in zip(epsilon_values, gaussian_errors, laplace_errors):
        print(f"{eps:<10.1f} {g_err:<20.4f} {l_err:<20.4f}")


def demonstrate_use_cases():
    """
    Demonstrate practical use cases for each mechanism
    """
    print("\n" + "=" * 80)
    print("Practical Use Cases")
    print("=" * 80)
    
    config = FSAConfig(
        n_features=10,
        num_clients=5,
        global_epochs=10,
        local_epochs=5,
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0
    )
    
    dp_tool = DifferentialPrivacy(config)
    
    print("\n1. Use Case: Counting Query (适用拉普拉斯机制)")
    print("-" * 80)
    true_count = 1000  # True number of patients
    count_tensor = torch.tensor([float(true_count)])
    noisy_count = dp_tool.add_laplace_noise(count_tensor, sensitivity=1.0, epsilon=1.0)
    print(f"True count: {true_count}")
    print(f"Noisy count: {int(noisy_count.item())}")
    print(f"Absolute error: {abs(noisy_count.item() - true_count):.2f}")
    
    print("\n2. Use Case: Model Gradient Update (适用高斯机制)")
    print("-" * 80)
    gradient = torch.randn(100)
    print(f"Original gradient norm: {torch.norm(gradient).item():.4f}")
    noisy_gradient = dp_tool.add_gaussian_noise(gradient.clone())
    print(f"Noisy gradient norm: {torch.norm(noisy_gradient).item():.4f}")
    print(f"Noise magnitude: {torch.norm(noisy_gradient - gradient).item():.4f}")
    
    print("\n3. Use Case: Best Model Selection (适用指数机制)")
    print("-" * 80)
    model_configs = torch.randn(5, 50)  # 5 candidate model configurations
    validation_scores = torch.tensor([0.82, 0.85, 0.88, 0.84, 0.86])  # Validation accuracy
    
    print("Candidate validation scores:", validation_scores.numpy())
    selected = dp_tool.exponential_mechanism(model_configs, validation_scores, epsilon=1.0)
    print(f"Selected model: {selected} (score: {validation_scores[selected]:.4f})")
    print(f"Best model: {torch.argmax(validation_scores).item()} (score: {validation_scores.max():.4f})")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run demonstrations
    demonstrate_noise_mechanisms()
    demonstrate_use_cases()
    
    print("\n" + "=" * 80)
    print("Summary: Three Differential Privacy Mechanisms")
    print("=" * 80)
    print("""
    ┌─────────────────────┬──────────────────────┬─────────────────────────┐
    │ Mechanism           │ Privacy Guarantee    │ Best Use Case           │
    ├─────────────────────┼──────────────────────┼─────────────────────────┤
    │ Gaussian (高斯)     │ (ε, δ)-DP           │ Deep Learning Gradients │
    │ Laplace (拉普拉斯)  │ ε-DP                │ Counting/Sum Queries    │
    │ Exponential (指数)  │ ε-DP                │ Non-numeric Selection   │
    └─────────────────────┴──────────────────────┴─────────────────────────┘
    
    Key Differences:
    - Gaussian: Normal distribution noise, requires δ parameter, best for ML
    - Laplace: Double exponential noise, pure ε-DP, best for queries
    - Exponential: Probabilistic selection, best for discrete choices
    """)
    
    print("\n✓ Example completed successfully!")
    print("=" * 80)
