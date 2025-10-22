"""
差分隐私功能测试用例
"""
import pytest
import torch
import numpy as np
import math
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy


@pytest.fixture
def dp_config():
    """创建差分隐私测试配置"""
    return FSAConfig(
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0
    )


@pytest.fixture
def dp_tool(dp_config):
    """创建差分隐私工具"""
    return DifferentialPrivacy(dp_config)


def test_initialization(dp_tool):
    """测试差分隐私工具初始化"""
    assert dp_tool.epsilon == 1.0
    assert dp_tool.delta == 1e-5
    assert dp_tool.sensitivity == 1.0
    assert dp_tool.noise_multiplier == 1.0
    assert dp_tool.clip_norm == 1.0


def test_add_gaussian_noise(dp_tool):
    """测试高斯噪声添加"""
    # 创建测试张量
    tensor = torch.ones(10, 5)
    original_tensor = tensor.clone()
    
    # 添加噪声
    noisy_tensor = dp_tool.add_gaussian_noise(tensor)
    
    # 验证噪声被添加
    assert not torch.equal(original_tensor, noisy_tensor)
    
    # 验证张量形状不变
    assert tensor.shape == noisy_tensor.shape
    
    # 验证噪声的统计特性
    noise = noisy_tensor - original_tensor
    noise_std = torch.std(noise)
    expected_std = dp_tool.sensitivity * dp_tool.noise_multiplier
    
    # 噪声标准差应该在期望值附近（允许一定误差）
    assert abs(noise_std.item() - expected_std) <= 0.5


def test_add_gaussian_noise_with_custom_sensitivity(dp_tool):
    """测试使用自定义敏感度的高斯噪声添加"""
    tensor = torch.ones(5, 3)
    custom_sensitivity = 2.0
    
    noisy_tensor = dp_tool.add_gaussian_noise(tensor, sensitivity=custom_sensitivity)
    
    # 验证噪声被添加
    assert not torch.equal(tensor, noisy_tensor)
    
    # 验证噪声规模
    noise = noisy_tensor - tensor
    noise_std = torch.std(noise)
    expected_std = custom_sensitivity * dp_tool.noise_multiplier
    
    assert abs(noise_std.item() - expected_std) <= 0.5


def test_clip_gradients(dp_tool):
    """测试梯度裁剪"""
    # 创建模拟模型
    model = torch.nn.Linear(5, 3)
    
    # 设置较大的梯度
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.ones_like(param.data) * 10.0  # 大梯度
    
    # 记录裁剪前的梯度范数
    total_norm_before = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** (1. / 2)
    
    # 应用梯度裁剪
    clipped_norm = dp_tool.clip_gradients(model)
    
    # 验证梯度被裁剪
    assert clipped_norm == total_norm_before
    
    # 验证裁剪后的梯度范数
    total_norm_after = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** (1. / 2)
    
    # 裁剪后的范数应该小于等于裁剪范数
    assert total_norm_after <= dp_tool.clip_norm + 1e-6


def test_clip_gradients_small_gradients(dp_tool):
    """测试小梯度的裁剪（不应该被裁剪）"""
    model = torch.nn.Linear(3, 2)
    
    # 设置小梯度
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.ones_like(param.data) * 0.1  # 小梯度
    
    # 应用梯度裁剪
    clipped_norm = dp_tool.clip_gradients(model)
    
    # 小梯度不应该被裁剪
    assert clipped_norm < dp_tool.clip_norm


def test_get_noise_scale(dp_tool):
    """测试噪声规模计算"""
    num_clients = 5
    noise_scale = dp_tool.get_noise_scale(num_clients)
    
    # 验证噪声规模计算
    expected_scale = math.sqrt(2 * math.log(1.25 / dp_tool.delta)) * dp_tool.sensitivity / dp_tool.epsilon
    expected_scale = expected_scale / math.sqrt(num_clients)
    
    assert abs(noise_scale - expected_scale) < 1e-6


def test_compute_privacy_budget(dp_tool):
    """测试隐私预算计算"""
    num_rounds = 10
    num_clients = 5
    
    total_epsilon, per_round_epsilon = dp_tool.compute_privacy_budget(num_rounds, num_clients)
    
    # 验证隐私预算计算
    assert total_epsilon == dp_tool.epsilon
    assert per_round_epsilon == dp_tool.epsilon / num_rounds


def test_apply_dp_to_weights(dp_tool):
    """测试权重差分隐私应用（已弃用方法）"""
    weights = {
        'weight': torch.ones(3, 2),
        'bias': torch.ones(2)
    }
    num_clients = 3
    
    # 应用差分隐私（应该返回原始权重）
    dp_weights = dp_tool.apply_dp_to_weights(weights, num_clients)
    
    # 验证返回原始权重
    for key in weights:
        assert torch.equal(weights[key], dp_weights[key])


def test_apply_dp_to_gradients(dp_tool):
    """测试梯度差分隐私应用"""
    model = torch.nn.Linear(4, 2)
    
    # 设置梯度
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.ones_like(param.data)
    
    # 记录原始梯度
    original_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            original_grads[name] = param.grad.clone()
    
    # 应用差分隐私
    grad_norm = dp_tool.apply_dp_to_gradients(model, torch.optim.Adam(model.parameters()))
    
    # 验证梯度被修改
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.equal(original_grads[name], param.grad)


def test_compute_renyi_divergence(dp_tool):
    """测试Renyi散度计算"""
    alpha = 2.0
    sigma = 1.0
    
    rdp = dp_tool.compute_renyi_divergence(alpha, sigma)
    
    # 验证Renyi散度计算
    expected_rdp = alpha / (2 * sigma ** 2)
    assert abs(rdp - expected_rdp) < 1e-6


def test_convert_renyi_to_epsilon(dp_tool):
    """测试Renyi差分隐私到ε转换"""
    alpha = 2.0
    rdp = 0.5
    
    epsilon = dp_tool.convert_renyi_to_epsilon(alpha, rdp)
    
    # 验证转换计算
    expected_epsilon = rdp + math.log(1 / dp_tool.delta) / (alpha - 1)
    assert abs(epsilon - expected_epsilon) < 1e-6


def test_noise_scale_with_different_clients(dp_tool):
    """测试不同客户端数量下的噪声规模"""
    scales = []
    for num_clients in [1, 5, 10, 20]:
        scale = dp_tool.get_noise_scale(num_clients)
        scales.append(scale)
    
    # 验证客户端数量越多，噪声规模越小
    for i in range(1, len(scales)):
        assert scales[i] < scales[i-1]


def test_privacy_parameters_validation():
    """测试隐私参数验证"""
    # 测试无效的ε值
    with pytest.raises(ValueError):
        FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=-1.0,  # 无效值
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
    
    # 测试无效的δ值
    with pytest.raises(ValueError):
        FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=1.0,
            dp_delta=1.5,  # 无效值，应该 < 1
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )


def test_differential_privacy_disabled():
    """测试差分隐私禁用时的行为"""
    config_no_dp = FSAConfig(use_differential_privacy=False)
    
    # 验证配置
    assert config_no_dp.use_differential_privacy is False
    
    # 验证不会抛出异常
    dp_tool = DifferentialPrivacy(config_no_dp)
    assert dp_tool is not None



def test_config_with_dp_parameters():
    """测试包含差分隐私参数的配置"""
    config = FSAConfig(
        n_samples=100,
        n_features=10,
        num_clients=3,
        use_differential_privacy=True,
        dp_epsilon=2.0,
        dp_delta=1e-6,
        dp_sensitivity=0.5,
        dp_noise_multiplier=1.5,
        dp_clip_norm=2.0
    )
    
    # 验证配置正确设置
    assert config.use_differential_privacy is True
    assert config.dp_epsilon == 2.0
    assert config.dp_delta == 1e-6
    assert config.dp_sensitivity == 0.5
    assert config.dp_noise_multiplier == 1.5
    assert config.dp_clip_norm == 2.0


def test_dp_tool_with_custom_config():
    """测试使用自定义配置的差分隐私工具"""
    config = FSAConfig(
        use_differential_privacy=True,
        dp_epsilon=0.5,
        dp_delta=1e-4,
        dp_sensitivity=2.0,
        dp_noise_multiplier=0.5,
        dp_clip_norm=0.5
    )
    
    dp_tool = DifferentialPrivacy(config)
    
    # 验证自定义参数
    assert dp_tool.epsilon == 0.5
    assert dp_tool.delta == 1e-4
    assert dp_tool.sensitivity == 2.0
    assert dp_tool.noise_multiplier == 0.5
    assert dp_tool.clip_norm == 0.5
    
    # 测试噪声添加
    tensor = torch.ones(5, 3)
    noisy_tensor = dp_tool.add_gaussian_noise(tensor)
    
    # 验证噪声被添加
    assert not torch.equal(tensor, noisy_tensor)
    
    # 验证噪声规模
    noise = noisy_tensor - tensor
    noise_std = torch.std(noise)
    expected_std = dp_tool.sensitivity * dp_tool.noise_multiplier
    
    assert abs(noise_std.item() - expected_std) <= 0.5
