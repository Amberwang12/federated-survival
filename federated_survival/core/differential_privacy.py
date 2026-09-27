"""
差分隐私工具模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import math


class DifferentialPrivacy:
    """差分隐私工具类
    
    注意：差分隐私噪声只在客户端本地训练时应用，
    不在服务器端模型聚合时添加噪声。
    """
    
    def __init__(self, config):
        """
        初始化差分隐私工具
        
        Args:
            config: 联邦学习配置
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
        添加高斯噪声实现差分隐私（高斯机制）
        
        添加由noise multiplier参数化的高斯噪声。
        噪声规模: σ = noise_multiplier × sensitivity。

        本方法只负责噪声机制。只有在被发布的查询/模型更新已经裁剪到
        相应敏感度，且对多轮发布进行隐私会计时，才能声称差分隐私保证。
        
        Args:
            tensor: 输入张量
            sensitivity: 敏感度，如果为None则使用配置中的值
            
        Returns:
            添加噪声后的张量
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
            
        sigma = self.noise_multiplier * sensitivity
        
        # 生成高斯噪声
        noise = torch.normal(0, sigma, size=tensor.shape, device=tensor.device, dtype=tensor.dtype)
        
        # 添加噪声
        return tensor + noise
    
    def add_laplace_noise(self, tensor: torch.Tensor, sensitivity: Optional[float] = None, epsilon: Optional[float] = None) -> torch.Tensor:
        """
        添加拉普拉斯噪声实现差分隐私（拉普拉斯机制）
        
        拉普拉斯机制提供ε-差分隐私保证，不需要δ参数。
        噪声规模: b = Δf / ε (Laplace分布的尺度参数)
        
        Args:
            tensor: 输入张量
            sensitivity: 敏感度，如果为None则使用配置中的值
            epsilon: 隐私预算，如果为None则使用配置中的值
            
        Returns:
            添加噪声后的张量
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        if epsilon is None:
            epsilon = self.epsilon
            
        # 计算拉普拉斯分布的尺度参数 b = Δf / ε
        scale = sensitivity / epsilon
        
        # 生成拉普拉斯噪声
        # PyTorch没有直接的Laplace分布，使用numpy生成后转换
        noise_np = np.random.laplace(loc=0.0, scale=scale, size=tensor.shape)
        noise = torch.from_numpy(noise_np).to(device=tensor.device, dtype=tensor.dtype)
        
        # 添加噪声
        return tensor + noise
    
    def exponential_mechanism(self, 
                            candidates: torch.Tensor, 
                            quality_scores: torch.Tensor, 
                            sensitivity: Optional[float] = None,
                            epsilon: Optional[float] = None) -> int:
        """
        指数机制实现差分隐私（指数机制）
        
        指数机制用于非数值输出的场景，通过概率采样选择候选项。
        选择概率: P(r) ∝ exp(ε·q(r) / (2·Δq))
        其中 q(r) 是候选项的质量得分，Δq 是质量函数的敏感度。
        
        Args:
            candidates: 候选项张量，形状为 (n_candidates, ...)
            quality_scores: 每个候选项的质量得分，形状为 (n_candidates,)
            sensitivity: 质量函数的敏感度，如果为None则使用配置中的值
            epsilon: 隐私预算，如果为None则使用配置中的值
            
        Returns:
            选中的候选项索引
            
        Example:
            >>> # 选择最优模型参数配置
            >>> candidates = torch.randn(10, 100)  # 10个候选配置
            >>> scores = torch.tensor([0.8, 0.85, 0.9, ...])  # 质量得分
            >>> selected_idx = dp.exponential_mechanism(candidates, scores)
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        if epsilon is None:
            epsilon = self.epsilon
            
        # 计算选择概率: P(r) ∝ exp(ε·q(r) / (2·Δq))
        scores = quality_scores.detach().cpu().numpy().astype(float)
        logits = epsilon * scores / (2 * sensitivity)
        logits -= logits.max()
        probabilities = np.exp(logits)
        
        # 归一化概率
        probabilities = probabilities / np.sum(probabilities)
        
        # 根据概率采样选择候选项
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        
        return selected_idx
    
    def exponential_mechanism_tensor(self,
                                    candidates: torch.Tensor,
                                    quality_scores: torch.Tensor,
                                    sensitivity: Optional[float] = None,
                                    epsilon: Optional[float] = None) -> torch.Tensor:
        """
        指数机制的张量返回版本
        
        Args:
            candidates: 候选项张量，形状为 (n_candidates, ...)
            quality_scores: 每个候选项的质量得分，形状为 (n_candidates,)
            sensitivity: 质量函数的敏感度
            epsilon: 隐私预算
            
        Returns:
            选中的候选项张量
        """
        selected_idx = self.exponential_mechanism(candidates, quality_scores, sensitivity, epsilon)
        return candidates[selected_idx]
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        裁剪梯度到指定范数
        
        Args:
            model: 模型
            
        Returns:
            裁剪前的梯度范数
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 裁剪梯度
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
                
        return total_norm
    
    def add_noise_to_weights(self, weights: Dict[str, torch.Tensor], num_clients: Optional[int] = 1) -> Dict[str, torch.Tensor]:
        """
        向模型权重添加差分隐私噪声
        
        Args:
            weights: 模型权重字典
            num_clients: 参与训练的客户端数量
            
        Returns:
            添加噪声后的权重字典
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
        计算隐私预算消耗
        
        Args:
            num_rounds: 训练轮数
            num_clients: 客户端数量
            
        Returns:
            (总隐私预算, 每轮隐私预算)
        """
        # 使用差分隐私的组成定理计算总隐私预算
        # 对于联邦学习，考虑每轮采样客户端的影响
        # 使用Ostrovsky与Rosen的组合定理近似计算
        
        if num_rounds <= 0 or num_clients <= 0:
            raise ValueError("num_rounds and num_clients must be positive")
        # 这是基本组合下的预算分配，不是从noise multiplier反推得到的会计值。
        per_round_epsilon = self.epsilon / num_rounds
        
        # 总隐私预算
        total_epsilon = self.epsilon
        
        return total_epsilon, per_round_epsilon
    
    def get_noise_scale(self, num_clients: int) -> float:
        """
        根据客户端数量计算噪声规模
        
        Args:
            num_clients: 参与训练的客户端数量
            
        Returns:
            噪声规模
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        # 独立客户端噪声经等权平均后的标准差。
        return self.noise_multiplier * self.sensitivity / math.sqrt(num_clients)
    
    def apply_dp_to_gradients(self, model: nn.Module, optimizer: torch.optim.Optimizer, mechanism: str = 'gaussian') -> float:
        """
        对梯度应用差分隐私保护
        
        Args:
            model: 模型
            optimizer: 优化器
            mechanism: 差分隐私机制，可选 'gaussian' 或 'laplace'
            
        Returns:
            裁剪前的梯度范数
        """
        # 裁剪梯度
        grad_norm = self.clip_gradients(model)
        
        # 根据机制选择添加不同类型的噪声
        for param in model.parameters():
            if param.grad is not None:
                if mechanism == 'gaussian':
                    # 高斯机制
                    noise = torch.normal(
                        0, 
                        self.get_noise_scale(num_clients=1),
                        size=param.grad.shape,
                        device=param.grad.device,
                        dtype=param.grad.dtype
                    )
                    param.grad.data.add_(other=noise)
                elif mechanism == 'laplace':
                    # 拉普拉斯机制
                    scale = self.sensitivity / self.epsilon
                    noise_np = np.random.laplace(loc=0.0, scale=scale, size=param.grad.shape)
                    noise = torch.from_numpy(noise_np).to(device=param.grad.device, dtype=param.grad.dtype)
                    param.grad.data.add_(other=noise)
                else:
                    raise ValueError(f"Unsupported mechanism: {mechanism}. Choose 'gaussian' or 'laplace'.")
        
        return grad_norm
    
    def apply_dp_to_weights(self, weights: Dict[str, torch.Tensor], num_clients: int) -> Dict[str, torch.Tensor]:
        """
        对聚合后的权重应用差分隐私保护
        注意：此方法已弃用，噪声现在只在客户端本地训练时添加
        
        Args:
            weights: 聚合后的权重
            num_clients: 参与聚合的客户端数量
            
        Returns:
            原始权重（不添加噪声）
        """
        # 返回原始权重，不添加噪声
        return weights
    
    def compute_renyi_divergence(self, alpha: float, sigma: float) -> float:
        """
        计算Renyi散度
        
        Args:
            alpha: Renyi散度的阶数
            sigma: 噪声标准差
            
        Returns:
            Renyi散度值
        """
        return alpha / (2 * sigma ** 2)
    
    def convert_renyi_to_epsilon(self, alpha: float, rdp: float) -> float:
        """
        将Renyi差分隐私转换为(ε, δ)-差分隐私
        
        Args:
            alpha: Renyi散度的阶数
            rdp: Renyi差分隐私参数
            
        Returns:
            ε值
        """
        return rdp + math.log(1 / self.delta) / (alpha - 1)
