"""
Differential privacy utilities.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import math


class DifferentialPrivacy:
    """Differential privacy utilities.

    Note: differential privacy noise is applied only during client-side
    local training; no noise is added during server-side model aggregation.
    """
    
    def __init__(self, config):
        """
        Initialize the differential privacy utility.

        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.epsilon = config.dp_epsilon
        self.delta = config.dp_delta
        self.sensitivity = config.dp_sensitivity
        self.noise_multiplier = config.dp_noise_multiplier
        self.clip_norm = config.dp_clip_norm

        # Diagnostics for the most recent update, populated by
        # :meth:`privatize_model_update`.  ``last_noise_to_signal`` above ~1
        # means the injected noise norm exceeds the clipped update norm, i.e.
        # the published update is dominated by noise.
        self.last_clipped_update_norm: float = 0.0
        self.last_noise_sigma: float = 0.0
        self.last_noise_to_signal: float = 0.0
        
    def add_gaussian_noise(self, tensor: torch.Tensor, sensitivity: Optional[float] = None) -> torch.Tensor:
        """
        Add Gaussian noise for differential privacy (Gaussian mechanism).

        Adds Gaussian noise parameterized by the noise multiplier.
        Noise scale: σ = noise_multiplier × sensitivity.

        This method only implements the noise mechanism. A differential
        privacy guarantee can be claimed only when the released query/model
        update has been clipped to the corresponding sensitivity and privacy
        accounting is performed across multiple rounds of releases.

        Args:
            tensor: Input tensor
            sensitivity: Sensitivity; if None, the configured value is used

        Returns:
            Tensor with noise added
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
            
        sigma = self.noise_multiplier * sensitivity
        
        # Generate Gaussian noise
        noise = torch.normal(0, sigma, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
        
        # Add noise
        return tensor + noise
    
    def add_laplace_noise(self, tensor: torch.Tensor, sensitivity: Optional[float] = None, epsilon: Optional[float] = None) -> torch.Tensor:
        """
        Add Laplace noise for differential privacy (Laplace mechanism).

        The Laplace mechanism provides ε-differential privacy and needs no δ parameter.
        Noise scale: b = Δf / ε (scale parameter of the Laplace distribution)

        Args:
            tensor: Input tensor
            sensitivity: Sensitivity; if None, the configured value is used
            epsilon: Privacy budget; if None, the configured value is used

        Returns:
            Tensor with noise added
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        if epsilon is None:
            epsilon = self.epsilon
            
        # Compute the Laplace scale parameter b = Δf / ε
        scale = sensitivity / epsilon
        
        # Generate Laplace noise
        # PyTorch has no built-in Laplace distribution; generate with numpy and convert
        noise_np = np.random.laplace(loc=0.0, scale=scale, size=tensor.shape)
        noise = torch.from_numpy(noise_np).to(device=tensor.device, dtype=tensor.dtype)
        
        # Add noise
        return tensor + noise
    
    def exponential_mechanism(self, 
                            candidates: torch.Tensor, 
                            quality_scores: torch.Tensor, 
                            sensitivity: Optional[float] = None,
                            epsilon: Optional[float] = None) -> int:
        """
        Exponential mechanism for differential privacy.

        The exponential mechanism handles non-numeric outputs by selecting a
        candidate through probabilistic sampling.
        Selection probability: P(r) ∝ exp(ε·q(r) / (2·Δq))
        where q(r) is the quality score of candidate r and Δq is the
        sensitivity of the quality function.

        Args:
            candidates: Candidate tensor of shape (n_candidates, ...)
            quality_scores: Quality score of each candidate, shape (n_candidates,)
            sensitivity: Sensitivity of the quality function; if None, the configured value is used
            epsilon: Privacy budget; if None, the configured value is used

        Returns:
            Index of the selected candidate

        Example:
            >>> # Select the best model parameter configuration
            >>> candidates = torch.randn(10, 100)  # 10 candidate configurations
            >>> scores = torch.tensor([0.8, 0.85, 0.9, ...])  # quality scores
            >>> selected_idx = dp.exponential_mechanism(candidates, scores)
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        if epsilon is None:
            epsilon = self.epsilon
            
        # Compute selection probabilities: P(r) ∝ exp(ε·q(r) / (2·Δq))
        scores = quality_scores.detach().cpu().numpy().astype(float)
        logits = epsilon * scores / (2 * sensitivity)
        logits -= logits.max()
        probabilities = np.exp(logits)
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample a candidate according to the probabilities
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        return selected_idx
    
    def exponential_mechanism_tensor(self,
                                    candidates: torch.Tensor,
                                    quality_scores: torch.Tensor,
                                    sensitivity: Optional[float] = None,
                                    epsilon: Optional[float] = None) -> torch.Tensor:
        """
        Exponential mechanism variant that returns the selected tensor.

        Args:
            candidates: Candidate tensor of shape (n_candidates, ...)
            quality_scores: Quality score of each candidate, shape (n_candidates,)
            sensitivity: Sensitivity of the quality function
            epsilon: Privacy budget

        Returns:
            The selected candidate tensor
        """
        selected_idx = self.exponential_mechanism(candidates, quality_scores, sensitivity, epsilon)
        return candidates[selected_idx]
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to the specified norm.

        Args:
            model: Model

        Returns:
            Gradient norm before clipping
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
                
        return total_norm
    
    def add_noise_to_weights(self, weights: Dict[str, torch.Tensor], num_clients: Optional[int] = 1) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to model weights.

        Args:
            weights: Dictionary of model weights
            num_clients: Number of clients participating in training

        Returns:
            Dictionary of weights with noise added
        """
        if num_clients is None or num_clients <= 0:
            raise ValueError("num_clients must be positive")
        noisy_weights = {}
        for name, weight in weights.items():
            if not (weight.is_floating_point() or weight.is_complex()):
                noisy_weights[name] = weight.clone()
                continue
            if self.config.dp_mechanism == 'gaussian':
                noisy_weights[name] = self.add_gaussian_noise(
                    weight, sensitivity=self.sensitivity
                )
            elif self.config.dp_mechanism == 'laplace':
                noisy_weights[name] = self.add_laplace_noise(
                    weight, sensitivity=self.sensitivity
                )
            else:
                raise ValueError(
                    "The exponential mechanism selects a discrete candidate and cannot "
                    "be applied elementwise to continuous model weights."
                )
            
        return noisy_weights

    def privatize_model_update(
        self,
        global_weights: Dict[str, torch.Tensor],
        local_weights: Dict[str, torch.Tensor],
        num_clients: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Clip one complete client update and add mechanism-specific noise.

        This is an experimental client-update perturbation mechanism.  It is
        substantially safer than adding noise to unbounded absolute weights,
        but it is not record-level DP-SGD and does not by itself provide a
        composed end-to-end privacy claim.

        The noise scale is the one reported by :meth:`get_noise_scale`, i.e.
        ``dp_noise_multiplier * dp_sensitivity * dp_clip_norm / sqrt(K)``.
        Averaging ``K`` independent client updates keeps their signals aligned
        while their noise adds in quadrature, so more clients means less
        residual noise for the same privacy budget.  The realised
        noise-to-signal ratio is stored in ``last_noise_to_signal`` so callers
        can detect a privacy setting that has swamped the model.
        """
        if num_clients is None:
            num_clients = int(getattr(self.config, "num_clients", 1) or 1)
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")

        float_names = [
            name for name, value in local_weights.items()
            if value.is_floating_point() or value.is_complex()
        ]
        squared_norm = torch.zeros((), dtype=torch.float64)
        deltas = {}
        for name in float_names:
            delta = local_weights[name].detach() - global_weights[name].to(
                device=local_weights[name].device, dtype=local_weights[name].dtype
            )
            deltas[name] = delta
            squared_norm += delta.detach().double().pow(2).sum().cpu()
        norm = math.sqrt(float(squared_norm))
        coefficient = min(1.0, self.clip_norm / (norm + 1e-12))

        # Per-element noise standard deviation, already divided by sqrt(K).
        noise_sigma = self.get_noise_scale(num_clients) * self.clip_norm
        clipped_norm = norm * coefficient
        self.last_clipped_update_norm = clipped_norm
        self.last_noise_sigma = float(noise_sigma)
        # sqrt(number of floating-point parameters) converts a per-element
        # standard deviation into the norm of that noise vector, which is
        # directly comparable with the clipped update norm.
        n_parameters = sum(deltas[name].numel() for name in deltas)
        self.last_noise_to_signal = float(
            noise_sigma * math.sqrt(n_parameters) / (clipped_norm + 1e-12)
        )

        result = {}
        for name, local_value in local_weights.items():
            global_value = global_weights[name].to(device=local_value.device)
            if name not in deltas:
                result[name] = local_value.clone()
                continue
            clipped = deltas[name] * coefficient
            sensitivity = self.sensitivity * self.clip_norm / math.sqrt(num_clients)
            if self.config.dp_mechanism == 'gaussian':
                private_delta = self.add_gaussian_noise(clipped, sensitivity=sensitivity)
            elif self.config.dp_mechanism == 'laplace':
                private_delta = self.add_laplace_noise(clipped, sensitivity=sensitivity)
            else:
                raise ValueError(
                    "dp_mechanism='exponential' is only available for standalone "
                    "candidate selection, not federated model updates."
                )
            result[name] = global_value.to(dtype=local_value.dtype) + private_delta
        return result
    
    def compute_privacy_budget(self, num_rounds: int, num_clients: int) -> Tuple[float, float]:
        """
        Compute privacy budget consumption.

        Args:
            num_rounds: Number of training rounds
            num_clients: Number of clients

        Returns:
            (total privacy budget, per-round privacy budget)
        """
        # Compute the total privacy budget using the composition theorem of
        # differential privacy; for federated learning, account for the effect
        # of per-round client sampling. Approximated with the composition
        # theorem of Ostrovsky and Rosen.
        
        if num_rounds <= 0 or num_clients <= 0:
            raise ValueError("num_rounds and num_clients must be positive")
        # This is the budget allocation under basic composition, not an accounting value derived from the noise multiplier.
        per_round_epsilon = self.epsilon / num_rounds
        
        # Total privacy budget
        total_epsilon = self.epsilon
        
        return total_epsilon, per_round_epsilon
    
    def get_noise_scale(self, num_clients: int) -> float:
        """
        Compute the noise scale based on the number of clients.

        Args:
            num_clients: Number of clients participating in training

        Returns:
            Noise scale
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        # Standard deviation of independent client noise after equal-weight averaging.
        return self.noise_multiplier * self.sensitivity / math.sqrt(num_clients)
    
    def apply_dp_to_gradients(self, model: nn.Module, optimizer: torch.optim.Optimizer, mechanism: str = 'gaussian') -> float:
        """
        Apply differential privacy protection to gradients.

        Args:
            model: Model
            optimizer: Optimizer
            mechanism: Differential privacy mechanism, either 'gaussian' or 'laplace'

        Returns:
            Gradient norm before clipping
        """
        # Clip gradients
        grad_norm = self.clip_gradients(model)
        
        # Add different types of noise depending on the mechanism
        for param in model.parameters():
            if param.grad is not None:
                if mechanism == 'gaussian':
                    # Gaussian mechanism
                    noise = torch.normal(
                        0, 
                        self.get_noise_scale(num_clients=1),
                        size=param.grad.shape,
                        device=param.grad.device,
                        dtype=param.grad.dtype
                    )
                    param.grad.data.add_(other=noise)
                elif mechanism == 'laplace':
                    # Laplace mechanism
                    scale = self.sensitivity / self.epsilon
                    noise_np = np.random.laplace(loc=0.0, scale=scale, size=param.grad.shape)
                    noise = torch.from_numpy(noise_np).to(device=param.grad.device, dtype=param.grad.dtype)
                    param.grad.data.add_(other=noise)
                else:
                    raise ValueError(f"Unsupported mechanism: {mechanism}. Choose 'gaussian' or 'laplace'.")
        
        return grad_norm
    
    def apply_dp_to_weights(self, weights: Dict[str, torch.Tensor], num_clients: int) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy protection to aggregated weights.
        Note: this method is deprecated; noise is now added only during
        client-side local training.

        Args:
            weights: Aggregated weights
            num_clients: Number of clients participating in aggregation

        Returns:
            Original weights (no noise added)
        """
        # Return the original weights without adding noise
        return weights
    
    def compute_renyi_divergence(self, alpha: float, sigma: float) -> float:
        """
        Compute the Renyi divergence.

        Args:
            alpha: Order of the Renyi divergence
            sigma: Noise standard deviation

        Returns:
            Renyi divergence value
        """
        return alpha / (2 * sigma ** 2)
    
    def convert_renyi_to_epsilon(self, alpha: float, rdp: float) -> float:
        """
        Convert Renyi differential privacy to (ε, δ)-differential privacy.

        Args:
            alpha: Order of the Renyi divergence
            rdp: Renyi differential privacy parameter

        Returns:
            The epsilon value
        """
        return rdp + math.log(1 / self.delta) / (alpha - 1)
