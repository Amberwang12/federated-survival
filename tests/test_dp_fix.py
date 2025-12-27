"""
测试差分隐私修复
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy
from federated_survival.data.generator import SimulationConfig, DataGenerator
from federated_survival.data.splitter import DataSplitter

def test_dp_basic_functionality():
    """测试基础差分隐私功能"""
    print("测试基础差分隐私功能...")
    
    # 创建配置
    config = FSAConfig(
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=1.0,
        dp_clip_norm=1.0
    )
    
    # 创建差分隐私工具
    dp_tool = DifferentialPrivacy(config)
    assert dp_tool.epsilon == 1.0, f"差分隐私工具创建失败"
    print(f"✅ 差分隐私工具创建成功: ε={dp_tool.epsilon}")
    
    # 测试噪声添加
    import torch
    tensor = torch.ones(5, 3)
    noisy_tensor = dp_tool.add_gaussian_noise(tensor)
    assert noisy_tensor.shape == tensor.shape, "噪声添加后形状不匹配"
    print(f"✅ 噪声添加成功: 原始形状={tensor.shape}, 噪声后形状={noisy_tensor.shape}")

def test_data_generation():
    """测试数据生成"""
    print("测试数据生成...")
    
    # 创建配置
    config = FSAConfig(
        n_samples=100,
        n_features=5,
        num_clients=3,
        censor_rate=0.3,
        random_seed=42
    )
    
    # 生成数据
    sim_config = SimulationConfig(
        n_samples=config.n_samples,
        n_features=config.n_features,
        random_state=config.random_seed
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull')
    assert raw_data is not None, "原始数据生成失败"
    print(f"✅ 原始数据生成成功: 形状={raw_data.shape}")
    
    # 转换为联邦学习格式
    splitter = DataSplitter(
        n_clients=config.num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=config.random_seed
    )
    federated_data = splitter.split(raw_data)
    assert len(federated_data.clients_set) == config.num_clients, "客户端数量不匹配"
    print(f"✅ 联邦学习数据生成成功: 客户端数量={len(federated_data.clients_set)}")

def test_dp_integration():
    """测试差分隐私集成"""
    print("测试差分隐私集成...")
    
    from federated_survival.core.runner import FSARunner
    
    # 创建配置
    config = FSAConfig(
        n_samples=50,  # 小数据集用于快速测试
        n_features=5,
        num_clients=2,
        global_epochs=1,
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
    
    # 生成数据
    sim_config = SimulationConfig(
        n_samples=config.n_samples,
        n_features=config.n_features,
        random_state=config.random_seed
    )
    generator = DataGenerator(sim_config)
    raw_data = generator.generate('weibull')
    
    splitter = DataSplitter(
        n_clients=config.num_clients,
        split_type='iid',
        test_size=0.2,
        random_state=config.random_seed
    )
    federated_data = splitter.split(raw_data)
    
    # 创建运行器
    runner = FSARunner(config)
    print(f"✅ 联邦学习运行器创建成功")
    
    # 获取隐私信息
    privacy_info = runner.get_privacy_info()
    assert privacy_info['privacy_protection'] is True, "隐私保护未启用"
    print(f"✅ 隐私信息获取成功: 隐私保护={privacy_info['privacy_protection']}")

def main():
    """主函数"""
    print("=" * 60)
    print("差分隐私修复验证测试")
    print("=" * 60)
    
    try:
        # 测试基础功能
        test_dp_basic_functionality()
        print()
        
        # 测试数据生成
        test_data_generation()
        print()
        
        # 测试集成
        test_dp_integration()
        print()
        
        print("✅ 所有测试通过！差分隐私功能修复成功。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
