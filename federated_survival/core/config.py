from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import torch.nn as nn

@dataclass
class FSAConfig:
    """联邦生存分析配置类"""
    # 数据参数
    dataset_name: Optional[str] = None  # 使用真实数据集时的名称
    mode: str = 'simulate'  # 可选: 'real', 'simulate'
    
    # 数据配置
    n_samples: int = 1000              # 样本数
    n_features: int = 20               # 特征数
    censor_rate: float = 0.4           # 删失率
    data_type: str = 'weibull'         # 可选: 'weibull', 'lognormal', '1', '2', '3', '4'
    sim_type: str = '1'                # 模拟数据类型
    
    # 模型参数
    model_type: str = 'PC-Hazard'      # 模型类型
    num_nodes: Tuple[int, ...] = (32, 32)    # 网络层数和节点数
    num_durations: int = 25            # 时间离散化数量
    batch_norm: bool = False           # 是否使用批标准化
    dropout: float = 0.1               # Dropout率
    activation: nn.Module = nn.ReLU    # 激活函数
    
    # 联邦学习参数
    num_clients: int = 5               # 客户端数量
    global_epochs: int = 50            # 全局训练轮数
    early_stopping: bool = False       # 是否使用早停
    early_stopping_patience: int = 10  # 早停的轮数
    local_epochs: int = 1              # 本地训练轮数
    batch_size: int = 32              # 批次大小
    learning_rate: float = 1e-3        # 学习率
    client_sample_ratio: float = 1.0   # 每轮选择的客户端比例
    split_method: str = 'iid'          # 数据划分方式，可选: 'iid', 'non-iid', 'time-non-iid'
    
    # 数据增强参数
    k: float = 0.5             # 增强数据比例
    latent_num: int = 10
    hidden_num: int = 30
    alpha: float = 1.0
    beta: float = 1.0
    
    # 差分隐私参数
    use_differential_privacy: bool = False  # 是否使用差分隐私
    dp_mechanism: str = 'gaussian'          # 差分隐私机制: 'gaussian', 'laplace', 'exponential'
    dp_epsilon: float = 1.0                 # 隐私预算 (ε)
    dp_delta: float = 1e-5                  # 失败概率 (δ) - 仅高斯机制需要
    dp_sensitivity: float = 1.0              # 敏感度
    dp_noise_multiplier: float = 1.0         # 噪声乘数 - 仅高斯机制使用
    dp_clip_norm: float = 1.0                # 梯度裁剪范数
    
    # 其他参数
    verbose: bool = False              # 是否打印详细信息
    random_seed: int = 42              # 随机种子
    
    # 基础配置
    n_rounds: int = 100
    random_state: Optional[int] = None
    
    # 模型配置
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # 数据划分配置
    split_alpha: float = 0.5  # 用于non-iid划分的狄利克雷分布参数
    test_size: float = 0.2

    
    def __post_init__(self):
        """验证配置参数的有效性"""
        # 验证模式
        valid_modes = ['real', 'simulate']
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        
        # 在real模式下需要提供数据集名称
        if self.mode == 'real' and self.dataset_name is None:
            raise ValueError("dataset_name must be provided when mode is 'real'")
        
        if not 0 <= self.censor_rate <= 1:
            raise ValueError("censor_rate must be between 0 and 1")
        
        if not 0 < self.client_sample_ratio <= 1:
            raise ValueError("client_sample_ratio must be between 0 and 1")
        
        # 验证模型类型
        valid_model_types = ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'CoxTime', 'DeepSurv', 'CoxPH', 'CoxCC']
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}")
        
        # 验证CoxPH的中间层参数
        if self.model_type == 'CoxPH':
            if self.num_nodes != ():
                raise ValueError("num_nodes must be () for CoxPH")

        # 验证数据划分方法
        valid_split_methods = ['iid', 'non-iid', 'time-non-iid']
        if self.split_method not in valid_split_methods:
            raise ValueError(f"split_method must be one of {valid_split_methods}")
        
        # 验证数值范围
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.k < 1:
            raise ValueError("k must be between 0 and 1")
        
        # 验证数据增强参数
        if self.latent_num <= 0:
            raise ValueError("latent_num must be positive")
        if self.hidden_num <= 0:
            raise ValueError("hidden_num must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        
        # 验证num_durations
        if self.num_durations <= 0:
            raise ValueError("num_durations must be positive")
        
        # 验证dropout
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        
        # 验证n_rounds
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive")
        
        # 验证split_alpha
        if self.split_alpha <= 0:
            raise ValueError("split_alpha must be positive")
        
        # 验证差分隐私参数
        if self.use_differential_privacy:
            # 验证机制类型
            valid_dp_mechanisms = ['gaussian', 'laplace', 'exponential']
            if self.dp_mechanism not in valid_dp_mechanisms:
                raise ValueError(f"dp_mechanism must be one of {valid_dp_mechanisms}")
            
            # 通用参数验证
            if self.dp_epsilon <= 0:
                raise ValueError("dp_epsilon must be positive")
            if self.dp_sensitivity <= 0:
                raise ValueError("dp_sensitivity must be positive")
            
            # 高斯机制特定参数
            if self.dp_mechanism == 'gaussian':
                if not 0 < self.dp_delta < 1:
                    raise ValueError("dp_delta must be between 0 and 1 for Gaussian mechanism")
                if self.dp_noise_multiplier <= 0:
                    raise ValueError("dp_noise_multiplier must be positive for Gaussian mechanism")
            
            # 梯度裁剪参数验证（适用于高斯和拉普拉斯机制）
            if self.dp_mechanism in ['gaussian', 'laplace']:
                if self.dp_clip_norm <= 0:
                    raise ValueError("dp_clip_norm must be positive")
        
        # 设置默认模型参数
        if not self.model_params:
            if self.model_type == 'PC-Hazard':
                self.model_params = {
                    'n_intervals': 10,
                    'hidden_size': 32,
                    'dropout': 0.1
                }
            elif self.model_type == 'LogisticHazard':
                self.model_params = {
                    'n_intervals': 10,
                    'hidden_size': 32,
                    'dropout': 0.1
                }
            elif self.model_type == 'DeepHit':
                self.model_params = {
                    'n_intervals': 10,
                    'hidden_size': 32,
                    'dropout': 0.1
                }
            elif self.model_type == 'CoxTime':
                self.model_params = {
                    'hidden_size': 32,
                    'dropout': 0.1
                }
            elif self.model_type in ['DeepSurv', 'CoxPH']:
                self.model_params = {
                    'l2_reg': 0.01
                }
            elif self.model_type == 'CoxCC':
                self.model_params = {
                    'l2_reg': 0.01
                }
        