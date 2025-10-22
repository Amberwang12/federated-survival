"""
Unit tests for the three differential privacy mechanisms:
1. Gaussian Mechanism
2. Laplace Mechanism
3. Exponential Mechanism

Author: Federated Survival Analysis Team
Date: 2025-10-19
"""

import unittest
import numpy as np
import torch
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.core.config import FSAConfig


class TestGaussianMechanism(unittest.TestCase):
    """Test Gaussian Mechanism"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FSAConfig(
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
        self.dp = DifferentialPrivacy(self.config)
    
    def test_gaussian_noise_shape(self):
        """Test that Gaussian noise preserves tensor shape"""
        tensor = torch.randn(100, 50)
        noisy_tensor = self.dp.add_gaussian_noise(tensor)
        
        self.assertEqual(tensor.shape, noisy_tensor.shape)
    
    def test_gaussian_noise_added(self):
        """Test that Gaussian noise is actually added"""
        tensor = torch.zeros(1000)
        noisy_tensor = self.dp.add_gaussian_noise(tensor)
        
        # Noise should be non-zero
        self.assertFalse(torch.allclose(tensor, noisy_tensor))
        
        # Mean of noise should be close to 0
        noise = noisy_tensor - tensor
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.5)
    
    def test_gaussian_noise_distribution(self):
        """Test that noise follows Gaussian distribution"""
        # Generate many samples
        n_samples = 10000
        tensor = torch.zeros(n_samples)
        noisy_tensor = self.dp.add_gaussian_noise(tensor)
        
        noise = (noisy_tensor - tensor).numpy()
        
        # Test normality using skewness and kurtosis
        from scipy import stats
        skewness = stats.skew(noise)
        kurtosis = stats.kurtosis(noise)
        
        # Gaussian has skewness~0 and excess kurtosis~0
        self.assertAlmostEqual(skewness, 0.0, delta=0.2)
        self.assertAlmostEqual(kurtosis, 0.0, delta=0.5)
    
    def test_gaussian_custom_sensitivity(self):
        """Test Gaussian mechanism with custom sensitivity"""
        tensor = torch.zeros(1000)
        
        # Higher sensitivity should lead to more noise
        noisy_low = self.dp.add_gaussian_noise(tensor.clone(), sensitivity=0.1)
        noisy_high = self.dp.add_gaussian_noise(tensor.clone(), sensitivity=10.0)
        
        noise_low = torch.std(noisy_low - tensor).item()
        noise_high = torch.std(noisy_high - tensor).item()
        
        self.assertGreater(noise_high, noise_low)


class TestLaplaceMechanism(unittest.TestCase):
    """Test Laplace Mechanism"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FSAConfig(
            n_features=10,
            num_clients=5,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_sensitivity=1.0
        )
        self.dp = DifferentialPrivacy(self.config)
    
    def test_laplace_noise_shape(self):
        """Test that Laplace noise preserves tensor shape"""
        tensor = torch.randn(100, 50)
        noisy_tensor = self.dp.add_laplace_noise(tensor)
        
        self.assertEqual(tensor.shape, noisy_tensor.shape)
    
    def test_laplace_noise_added(self):
        """Test that Laplace noise is actually added"""
        tensor = torch.zeros(1000)
        noisy_tensor = self.dp.add_laplace_noise(tensor)
        
        # Noise should be non-zero
        self.assertFalse(torch.allclose(tensor, noisy_tensor))
        
        # Mean of noise should be close to 0
        noise = noisy_tensor - tensor
        self.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.5)
    
    def test_laplace_noise_distribution(self):
        """Test that noise follows Laplace distribution"""
        # Generate many samples
        n_samples = 10000
        tensor = torch.zeros(n_samples)
        noisy_tensor = self.dp.add_laplace_noise(tensor, epsilon=1.0)
        
        noise = (noisy_tensor - tensor).numpy()
        
        # Laplace distribution has zero skewness and excess kurtosis of 3
        from scipy import stats
        skewness = stats.skew(noise)
        kurtosis = stats.kurtosis(noise)
        
        self.assertAlmostEqual(skewness, 0.0, delta=0.2)
        self.assertAlmostEqual(kurtosis, 3.0, delta=1.0)
    
    def test_laplace_epsilon_effect(self):
        """Test that smaller epsilon leads to more noise"""
        tensor = torch.zeros(1000)
        
        # Smaller epsilon should lead to more noise
        noisy_small_eps = self.dp.add_laplace_noise(tensor.clone(), epsilon=0.1)
        noisy_large_eps = self.dp.add_laplace_noise(tensor.clone(), epsilon=10.0)
        
        noise_small = torch.std(noisy_small_eps - tensor).item()
        noise_large = torch.std(noisy_large_eps - tensor).item()
        
        self.assertGreater(noise_small, noise_large)
    
    def test_laplace_vs_gaussian(self):
        """Test that Laplace has heavier tails than Gaussian"""
        n_samples = 10000
        tensor = torch.zeros(n_samples)
        
        # Generate noise
        laplace_noisy = self.dp.add_laplace_noise(tensor.clone(), epsilon=1.0)
        gaussian_noisy = self.dp.add_gaussian_noise(tensor.clone(), sensitivity=1.0)
        
        laplace_noise = (laplace_noisy - tensor).numpy()
        gaussian_noise = (gaussian_noisy - tensor).numpy()
        
        # Laplace should have higher kurtosis (heavier tails)
        from scipy import stats
        laplace_kurtosis = stats.kurtosis(laplace_noise)
        gaussian_kurtosis = stats.kurtosis(gaussian_noise)
        
        self.assertGreater(laplace_kurtosis, gaussian_kurtosis)


class TestExponentialMechanism(unittest.TestCase):
    """Test Exponential Mechanism"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FSAConfig(
            n_features=10,
            num_clients=5,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_sensitivity=1.0
        )
        self.dp = DifferentialPrivacy(self.config)
    
    def test_exponential_selection_range(self):
        """Test that exponential mechanism selects valid index"""
        candidates = torch.randn(10, 100)
        scores = torch.rand(10)
        
        selected_idx = self.dp.exponential_mechanism(candidates, scores)
        
        self.assertGreaterEqual(selected_idx, 0)
        self.assertLess(selected_idx, len(candidates))
    
    def test_exponential_favors_higher_scores(self):
        """Test that higher scores are selected more often"""
        candidates = torch.randn(5, 10)
        scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.4])  # Index 3 has highest score
        
        # Run multiple trials
        n_trials = 1000
        selection_counts = np.zeros(len(scores))
        
        for _ in range(n_trials):
            idx = self.dp.exponential_mechanism(candidates, scores, epsilon=2.0)
            selection_counts[idx] += 1
        
        # Highest score should be selected most often
        self.assertEqual(np.argmax(selection_counts), 3)
        
        # Highest score should be selected significantly more than lowest
        self.assertGreater(selection_counts[3], selection_counts[0])
    
    def test_exponential_epsilon_effect(self):
        """Test that higher epsilon makes selection more deterministic"""
        candidates = torch.randn(5, 10)
        scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.4])
        
        n_trials = 500
        
        # Low epsilon (more random)
        counts_low_eps = np.zeros(len(scores))
        for _ in range(n_trials):
            idx = self.dp.exponential_mechanism(candidates, scores, epsilon=0.1)
            counts_low_eps[idx] += 1
        
        # High epsilon (more deterministic)
        counts_high_eps = np.zeros(len(scores))
        for _ in range(n_trials):
            idx = self.dp.exponential_mechanism(candidates, scores, epsilon=10.0)
            counts_high_eps[idx] += 1
        
        # High epsilon should concentrate more on best candidate
        best_idx = np.argmax(scores.numpy())
        self.assertGreater(
            counts_high_eps[best_idx] / n_trials,
            counts_low_eps[best_idx] / n_trials
        )
    
    def test_exponential_tensor_return(self):
        """Test tensor return version of exponential mechanism"""
        candidates = torch.randn(5, 100)
        scores = torch.rand(5)
        
        selected = self.dp.exponential_mechanism_tensor(candidates, scores)
        
        # Should return one of the candidates
        self.assertEqual(selected.shape, candidates[0].shape)
        
        # Should match one of the candidates
        found = False
        for i in range(len(candidates)):
            if torch.allclose(selected, candidates[i]):
                found = True
                break
        self.assertTrue(found)
    
    def test_exponential_uniform_scores(self):
        """Test that uniform scores lead to uniform selection"""
        candidates = torch.randn(5, 10)
        scores = torch.ones(5)  # All equal scores
        
        n_trials = 1000
        selection_counts = np.zeros(len(scores))
        
        for _ in range(n_trials):
            idx = self.dp.exponential_mechanism(candidates, scores, epsilon=1.0)
            selection_counts[idx] += 1
        
        # All candidates should be selected roughly equally
        expected_count = n_trials / len(scores)
        for count in selection_counts:
            self.assertAlmostEqual(count, expected_count, delta=expected_count * 0.3)


class TestMechanismComparison(unittest.TestCase):
    """Test comparisons between mechanisms"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FSAConfig(
            n_features=10,
            num_clients=5,
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0
        )
        self.dp = DifferentialPrivacy(self.config)
    
    def test_gaussian_vs_laplace_noise_magnitude(self):
        """Compare noise magnitude between Gaussian and Laplace"""
        n_samples = 5000
        tensor = torch.zeros(n_samples)
        
        # Generate noise multiple times and average
        n_trials = 10
        gaussian_errors = []
        laplace_errors = []
        
        for _ in range(n_trials):
            gaussian_noisy = self.dp.add_gaussian_noise(tensor.clone())
            laplace_noisy = self.dp.add_laplace_noise(tensor.clone(), epsilon=1.0)
            
            gaussian_errors.append(torch.mean(torch.abs(gaussian_noisy - tensor)).item())
            laplace_errors.append(torch.mean(torch.abs(laplace_noisy - tensor)).item())
        
        avg_gaussian_error = np.mean(gaussian_errors)
        avg_laplace_error = np.mean(laplace_errors)
        
        # Both should add noise (non-zero error)
        self.assertGreater(avg_gaussian_error, 0.1)
        self.assertGreater(avg_laplace_error, 0.1)
    
    def test_all_mechanisms_preserve_privacy(self):
        """Test that all mechanisms provide privacy protection"""
        # Original data
        data = torch.randn(100)
        
        # Apply all mechanisms
        gaussian_noisy = self.dp.add_gaussian_noise(data.clone())
        laplace_noisy = self.dp.add_laplace_noise(data.clone())
        
        # All should differ from original
        self.assertFalse(torch.allclose(data, gaussian_noisy))
        self.assertFalse(torch.allclose(data, laplace_noisy))
        
        # Exponential mechanism (discrete selection)
        candidates = data.unsqueeze(0).repeat(5, 1)
        scores = torch.rand(5)
        selected_idx = self.dp.exponential_mechanism(candidates, scores)
        self.assertIsInstance(selected_idx, (int, np.integer))


class TestPrivacyBudgetAccounting(unittest.TestCase):
    """Test privacy budget accounting"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = FSAConfig(
            n_features=10,
            num_clients=5,
            use_differential_privacy=True,
            dp_epsilon=10.0,
            dp_delta=1e-5,
            dp_sensitivity=1.0
        )
        self.dp = DifferentialPrivacy(self.config)
    
    def test_privacy_budget_computation(self):
        """Test privacy budget computation"""
        total_eps, per_round_eps = self.dp.compute_privacy_budget(
            num_rounds=100,
            num_clients=5
        )
        
        self.assertAlmostEqual(total_eps, self.config.dp_epsilon)
        self.assertAlmostEqual(per_round_eps, self.config.dp_epsilon / 100)
    
    def test_noise_scale_calculation(self):
        """Test noise scale calculation"""
        scale_5_clients = self.dp.get_noise_scale(num_clients=5)
        scale_10_clients = self.dp.get_noise_scale(num_clients=10)
        
        # More clients should lead to less noise per client
        self.assertLess(scale_10_clients, scale_5_clients)


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run tests
    unittest.main(verbosity=2)
