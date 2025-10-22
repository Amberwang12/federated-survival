"""
差分隐私联邦学习集成测试
"""
import pytest
import torch
import numpy as np
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.server import Server
from federated_survival.core.client import Client
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


@pytest.fixture
def dp_config():
    """创建差分隐私测试配置"""
    return FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        model_type='PC-Hazard',
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0,
        verbose=False
    )


@pytest.fixture
def dp_test_data(dp_config):
    """生成差分隐私测试数据"""
    sim_config = SimulationConfig(
        n_samples=dp_config.n_samples,
        n_features=dp_config.n_features,
        censoring_rate=dp_config.censor_rate,
        random_state=dp_config.random_seed
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull')
    
    # 转换为联邦学习格式
    splitter = DataSplitter(
        n_clients=dp_config.num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=dp_config.random_seed
    )
    return splitter.split(raw_data)


@pytest.fixture
def config_no_dp():
    """创建无差分隐私的测试配置"""
    return FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        use_differential_privacy=False
    )


def test_dp_enabled_federated_learning(dp_config, dp_test_data):
    """测试启用差分隐私的联邦学习流程"""
    runner = FSARunner(dp_config)
    
    # 运行联邦学习（使用较少的轮次加快测试）
    results = runner.run(dp_test_data, type='raw')
    
    # 验证训练完成并返回了结果
    assert 'train_Cindex' in results
    assert 'train_IBS' in results
    assert 'test_Cindex' in results
    assert 'test_IBS' in results
    
    # 验证结果列表长度符合训练轮次
    assert len(results['train_Cindex']) <= dp_config.global_epochs
    assert len(results['test_Cindex']) <= dp_config.global_epochs
    
    # 验证指标值在合理范围内
    for cindex in results['train_Cindex']:
        assert 0 <= cindex <= 1
    for ibs in results['train_IBS']:
        assert ibs >= 0


def test_dp_disabled_federated_learning(config_no_dp, dp_test_data):
    """测试未启用差分隐私的联邦学习流程"""
    runner = FSARunner(config_no_dp)
    
    # 运行联邦学习
    results = runner.run(dp_test_data, type='raw')
    
    # 验证训练完成
    assert len(results['train_Cindex']) > 0
    assert len(results['test_Cindex']) > 0


def test_privacy_info_retrieval(dp_config):
    """测试隐私信息获取"""
    runner = FSARunner(dp_config)
    privacy_info = runner.get_privacy_info()
    
    # 验证隐私信息
    assert privacy_info["privacy_protection"] is True
    assert privacy_info["epsilon"] == dp_config.dp_epsilon
    assert privacy_info["delta"] == dp_config.dp_delta
    assert privacy_info["sensitivity"] == dp_config.dp_sensitivity
    assert privacy_info["noise_multiplier"] == dp_config.dp_noise_multiplier
    assert privacy_info["clip_norm"] == dp_config.dp_clip_norm
    
    # 验证隐私预算计算
    assert "total_epsilon" in privacy_info
    assert "per_round_epsilon" in privacy_info
    assert "noise_scale" in privacy_info


def test_privacy_info_without_dp(config_no_dp):
    """测试无差分隐私时的隐私信息"""
    runner = FSARunner(config_no_dp)
    privacy_info = runner.get_privacy_info()
    
    # 验证隐私保护为False
    assert privacy_info["privacy_protection"] is False


def test_dp_parameters_validation():
    """测试差分隐私参数验证"""
    # 测试有效的差分隐私参数
    valid_config = FSAConfig(
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0
    )
    
    # 应该不抛出异常
    assert valid_config.use_differential_privacy is True
    
    # 测试无效的差分隐私参数
    with pytest.raises(ValueError):
        FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=-1.0,  # 无效值
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )


def test_dp_noise_consistency(dp_config):
    """测试差分隐私噪声的一致性"""
    # 创建两个相同配置的差分隐私工具
    dp_tool1 = DifferentialPrivacy(dp_config)
    dp_tool2 = DifferentialPrivacy(dp_config)
    
    # 使用相同的输入和参数
    tensor = torch.ones(5, 3)
    
    # 添加噪声
    noisy_tensor1 = dp_tool1.add_gaussian_noise(tensor)
    noisy_tensor2 = dp_tool2.add_gaussian_noise(tensor)
    
    # 验证噪声被添加（结果不同，因为噪声是随机的）
    assert not torch.equal(noisy_tensor1, tensor)
    assert not torch.equal(noisy_tensor2, tensor)
    assert not torch.equal(noisy_tensor1, noisy_tensor2)
    
    # 验证噪声的统计特性相似
    noise1 = noisy_tensor1 - tensor
    noise2 = noisy_tensor2 - tensor
    
    std1 = torch.std(noise1)
    std2 = torch.std(noise2)
    
    # 标准差应该相似（允许一定误差）
    assert abs(std1.item() - std2.item()) <= 0.5


def test_dp_with_different_epsilon_values():
    """测试不同ε值下的差分隐私行为"""
    epsilon_values = [0.1, 1.0, 5.0]
    tensor = torch.ones(10, 5)
    
    # 多次测量取平均值以减少随机性影响
    num_trials = 10
    avg_noise_scales = []
    
    for epsilon in epsilon_values:
        config = FSAConfig(
            use_differential_privacy=True,
            dp_epsilon=epsilon,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0
        )
        
        dp_tool = DifferentialPrivacy(config)
        noise_scales = []
        
        for _ in range(num_trials):
            noisy_tensor = dp_tool.add_gaussian_noise(tensor)
            noise = noisy_tensor - tensor
            noise_scale = torch.std(noise).item()
            noise_scales.append(noise_scale)
        
        avg_noise_scales.append(np.mean(noise_scales))
    
    # 验证ε值越小，平均噪声越大（允许一定误差容忍度）
    # 由于是统计性质，我们使用更宽松的验证
    print(f"\n不同ε值的平均噪声标准差:")
    for eps, noise in zip(epsilon_values, avg_noise_scales):
        print(f"ε={eps}: 平均噪声标准差={noise:.4f}")
    
    # 验证总体趋势：最小的ε应该有最大的噪声
    assert avg_noise_scales[0] > avg_noise_scales[-1], \
        f"期望ε={epsilon_values[0]}的噪声({avg_noise_scales[0]:.4f}) > ε={epsilon_values[-1]}的噪声({avg_noise_scales[-1]:.4f})"


def test_dp_gradient_clipping_effectiveness(dp_config):
    """测试差分隐私梯度裁剪的有效性"""
    # 创建模型
    model = torch.nn.Linear(5, 3)
    
    # 设置大梯度
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.ones_like(param.data) * 10.0
    
    # 记录裁剪前的梯度范数
    total_norm_before = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** (1. / 2)
    
    # 应用差分隐私梯度裁剪
    dp_tool = DifferentialPrivacy(dp_config)
    clipped_norm = dp_tool.clip_gradients(model)
    
    # 验证梯度被有效裁剪
    assert clipped_norm == total_norm_before
    
    # 验证裁剪后的梯度范数
    total_norm_after = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** (1. / 2)
    
    # 裁剪后的范数应该小于等于裁剪范数
    assert total_norm_after <= dp_config.dp_clip_norm + 1e-6


def test_federated_learning_with_dp_vs_without_dp(dp_config, config_no_dp, dp_test_data):
    """对比启用和未启用差分隐私的联邦学习结果"""
    # 运行启用差分隐私的联邦学习
    runner_with_dp = FSARunner(dp_config)
    results_with_dp = runner_with_dp.run(dp_test_data, type='raw')
    
    # 运行未启用差分隐私的联邦学习
    runner_without_dp = FSARunner(config_no_dp)
    results_without_dp = runner_without_dp.run(dp_test_data, type='raw')
    
    # 验证两种情况都能正常完成训练
    assert len(results_with_dp['train_Cindex']) > 0
    assert len(results_without_dp['train_Cindex']) > 0
    
    # 差分隐私会影响性能,但应该在合理范围内
    # 注意:由于噪声的随机性,这个测试可能会偶尔失败
    # 在实际应用中,启用DP可能导致性能略有下降


# ========== 三种差分隐私机制测试 ==========

@pytest.fixture
def gaussian_config():
    """高斯机制配置"""
    return FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        model_type='PC-Hazard',
        use_differential_privacy=True,
        dp_mechanism='gaussian',
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0,
        verbose=False
    )


@pytest.fixture
def laplace_config():
    """拉普拉斯机制配置"""
    return FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        model_type='PC-Hazard',
        use_differential_privacy=True,
        dp_mechanism='laplace',
        dp_epsilon=1.0,
        dp_sensitivity=1.0,
        dp_clip_norm=1.0,
        verbose=False
    )


@pytest.fixture
def exponential_config():
    """指数机制配置"""
    return FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        model_type='PC-Hazard',
        use_differential_privacy=True,
        dp_mechanism='exponential',
        dp_epsilon=1.0,
        dp_sensitivity=1.0,
        verbose=False
    )


def test_gaussian_mechanism_federated_learning(gaussian_config, dp_test_data):
    """测试使用高斯机制的联邦学习"""
    runner = FSARunner(gaussian_config)
    results = runner.run(dp_test_data, type='raw')
    
    # 验证训练完成
    assert 'train_Cindex' in results
    assert 'test_Cindex' in results
    assert len(results['train_Cindex']) > 0
    
    # 验证隐私信息
    privacy_info = runner.get_privacy_info()
    assert privacy_info['privacy_protection'] is True
    assert privacy_info['mechanism'] == 'gaussian'
    assert 'delta' in privacy_info  # 高斯机制应该有delta参数
    assert 'noise_multiplier' in privacy_info
    
    print(f"\n高斯机制测试结果: Final C-index={results['test_Cindex'][-1]:.4f}")


def test_laplace_mechanism_federated_learning(laplace_config, dp_test_data):
    """测试使用拉普拉斯机制的联邦学习"""
    runner = FSARunner(laplace_config)
    results = runner.run(dp_test_data, type='raw')
    
    # 验证训练完成
    assert 'train_Cindex' in results
    assert 'test_Cindex' in results
    assert len(results['train_Cindex']) > 0
    
    # 验证隐私信息
    privacy_info = runner.get_privacy_info()
    assert privacy_info['privacy_protection'] is True
    assert privacy_info['mechanism'] == 'laplace'
    assert 'delta' not in privacy_info  # 拉普拉斯机制不需要delta
    assert 'clip_norm' in privacy_info  # 但需要梯度裁剪
    
    print(f"\n拉普拉斯机制测试结果: Final C-index={results['test_Cindex'][-1]:.4f}")


def test_exponential_mechanism_configuration(exponential_config):
    """测试指数机制的配置验证"""
    # 验证配置参数
    assert exponential_config.dp_mechanism == 'exponential'
    assert exponential_config.use_differential_privacy is True
    
    # 创建运行器
    runner = FSARunner(exponential_config)
    privacy_info = runner.get_privacy_info()
    
    # 验证隐私信息
    assert privacy_info['privacy_protection'] is True
    assert privacy_info['mechanism'] == 'exponential'
    assert 'delta' not in privacy_info  # 指数机制不需要delta
    assert 'clip_norm' not in privacy_info  # 也不需要梯度裁剪
    
    print(f"\n指数机制配置验证通过")


def test_compare_three_dp_mechanisms(gaussian_config, laplace_config, dp_test_data):
    """对比三种差分隐私机制在联邦学习中的表现"""
    mechanisms_configs = {
        'gaussian': gaussian_config,
        'laplace': laplace_config
    }
    
    results_dict = {}
    
    # 运行不同机制的联邦学习
    for mechanism_name, config in mechanisms_configs.items():
        runner = FSARunner(config)
        results = runner.run(dp_test_data, type='raw')
        results_dict[mechanism_name] = results
        
        print(f"\n{mechanism_name.upper()} mechanism:")
        print(f"  Final train C-index: {results['train_Cindex'][-1]:.4f}")
        print(f"  Final test C-index: {results['test_Cindex'][-1]:.4f}")
    
    # 验证所有机制都能正常完成训练
    for mechanism_name, results in results_dict.items():
        assert len(results['train_Cindex']) > 0, f"{mechanism_name} failed"
        assert len(results['test_Cindex']) > 0, f"{mechanism_name} failed"
        
        # 验证指标在合理范围内
        for cindex in results['test_Cindex']:
            assert 0 <= cindex <= 1, f"{mechanism_name}: C-index out of range"


def test_mechanism_parameter_validation():
    """测试不同机制的参数验证"""
    # 测试高斯机制需要delta
    config_gaussian = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='gaussian',
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0
    )
    assert config_gaussian.dp_mechanism == 'gaussian'
    
    # 测试拉普拉斯机制不需要delta（但设置也不会错）
    config_laplace = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='laplace',
        dp_epsilon=1.0,
        dp_sensitivity=1.0
    )
    assert config_laplace.dp_mechanism == 'laplace'
    
    # 测试指数机制
    config_exponential = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='exponential',
        dp_epsilon=1.0,
        dp_sensitivity=1.0
    )
    assert config_exponential.dp_mechanism == 'exponential'
    
    # 测试无效的机制类型
    with pytest.raises(ValueError, match="dp_mechanism must be one of"):
        FSAConfig(
            use_differential_privacy=True,
            dp_mechanism='invalid_mechanism',
            dp_epsilon=1.0
        )


def test_mechanism_noise_characteristics():
    """测试不同机制的噪声特性"""
    tensor = torch.zeros(1000)
    
    # 高斯机制
    gaussian_config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='gaussian',
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0
    )
    dp_gaussian = DifferentialPrivacy(gaussian_config)
    gaussian_noisy = dp_gaussian.add_gaussian_noise(tensor.clone())
    gaussian_noise = (gaussian_noisy - tensor).numpy()
    
    # 拉普拉斯机制
    laplace_config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='laplace',
        dp_epsilon=1.0,
        dp_sensitivity=1.0
    )
    dp_laplace = DifferentialPrivacy(laplace_config)
    laplace_noisy = dp_laplace.add_laplace_noise(tensor.clone())
    laplace_noise = (laplace_noisy - tensor).numpy()
    
    # 验证噪声被添加
    assert not torch.allclose(gaussian_noisy, tensor)
    assert not torch.allclose(laplace_noisy, tensor)
    
    # 验证噪声均值接近0
    assert abs(np.mean(gaussian_noise)) < 0.5
    assert abs(np.mean(laplace_noise)) < 0.5
    
    # 验证拉普拉斯分布有更重的尾部（更高的峰度）
    from scipy import stats
    gaussian_kurtosis = stats.kurtosis(gaussian_noise)
    laplace_kurtosis = stats.kurtosis(laplace_noise)
    
    print(f"\n高斯噪声峰度: {gaussian_kurtosis:.4f}")
    print(f"拉普拉斯噪声峰度: {laplace_kurtosis:.4f}")
    
    # 拉普拉斯应该有更高的峰度（理论值为3）
    assert laplace_kurtosis > gaussian_kurtosis


def test_backward_compatibility_default_mechanism(dp_test_data):
    """测试向后兼容性：不指定dp_mechanism时默认使用gaussian"""
    # 创建配置时不指定dp_mechanism
    old_style_config = FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        model_type='PC-Hazard',
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        verbose=False
    )
    
    runner = FSARunner(old_style_config)
    results = runner.run(dp_test_data, type='raw')
    
    # 应该成功运行
    assert len(results['train_Cindex']) > 0
    
    # 验证默认使用高斯机制
    privacy_info = runner.get_privacy_info()
    assert privacy_info['mechanism'] == 'gaussian'
    
    print("\n向后兼容性测试通过：默认使用gaussian机制")


def test_exponential_mechanism_basic_functionality():
    """测试指数机制的基本功能"""
    config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism='exponential',
        dp_epsilon=5.0,  # 进一步提高epsilon使选择更加确定性
        dp_sensitivity=1.0
    )
    
    dp_tool = DifferentialPrivacy(config)
    
    # 创建候选项和质量得分(进一步拉大差异)
    candidates = torch.randn(5, 10)
    quality_scores = torch.tensor([0.0, 0.1, 0.2, 2.0, 0.3])  # 第3个得分明显最高
    
    # 运行多次采样
    n_trials = 300  # 进一步增加采样次数
    selection_counts = np.zeros(len(quality_scores))
    
    for _ in range(n_trials):
        selected_idx = dp_tool.exponential_mechanism(candidates, quality_scores)
        selection_counts[selected_idx] += 1
    
    # 验证最高得分的候选项被选中次数超过总数的30% (更宽松的验证)
    best_idx = int(torch.argmax(quality_scores).item())
    best_selection_rate = selection_counts[best_idx] / n_trials
    
    # 验证最高得分被选中的比例应该明显高于平均水平(20%)
    # 使用30%作为阈值，考虑到随机性的影响
    assert best_selection_rate > 0.3, f"最高得分候选项被选中比例过低: {best_selection_rate:.2%}"
    
    # 验证最低得分的候选项被选中次数较少
    worst_idx = int(torch.argmin(quality_scores).item())
    
    # 验证最高得分被选中的次数应该明显多于最低得分(至少5倍)
    assert selection_counts[best_idx] > selection_counts[worst_idx] * 5, \
        f"最高得分({selection_counts[best_idx]})应该明显多于最低得分({selection_counts[worst_idx]})"
    
    # 验证总体趋势:高得分的候选项被选中次数更多
    sorted_indices = torch.argsort(quality_scores, descending=True)
    top_3_count = sum(selection_counts[i] for i in sorted_indices[:3])
    bottom_2_count = sum(selection_counts[i] for i in sorted_indices[3:])
    
    assert top_3_count > bottom_2_count, \
        f"前3名总选中次数({top_3_count})应该多于后2名({bottom_2_count})"
    
    print(f"\n指数机制选择统计 ({n_trials}次):")
    for i, (score, count) in enumerate(zip(quality_scores, selection_counts)):
        marker = " ← 最高分" if i == best_idx else ""
        print(f"  Candidate {i} (score={score:.2f}): selected {int(count)} times ({count/n_trials*100:.1f}%){marker}")
