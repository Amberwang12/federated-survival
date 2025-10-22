import pytest
import torch.nn as nn
from federated_survival.core.config import FSAConfig

def test_config_initialization():
    """测试配置类的初始化"""
    config = FSAConfig()
    # 数据参数
    assert config.dataset_name is None
    assert config.mode == 'simulate'
    assert config.n_samples == 1000
    assert config.n_features == 20
    assert config.censor_rate == 0.4
    assert config.data_type == 'weibull'
    assert config.sim_type == '1'
    
    # 模型参数
    assert config.model_type == 'PC-Hazard'
    assert config.num_nodes == (32, 32)
    assert config.num_durations == 25
    assert config.batch_norm is False
    assert config.dropout == 0.1
    assert config.activation == nn.ReLU  # 检查类而不是实例
    
    # 联邦学习参数
    assert config.num_clients == 5
    assert config.global_epochs == 50
    assert config.early_stopping is False
    assert config.early_stopping_patience == 10
    assert config.local_epochs == 1
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.client_sample_ratio == 1.0
    assert config.split_method == 'iid'
    
    # 数据增强参数
    assert config.k == 0.5
    assert config.latent_num == 10
    assert config.hidden_num == 30
    assert config.alpha == 1.0
    assert config.beta == 1.0
    
    # 其他参数
    assert config.verbose is False
    assert config.random_seed == 42
    
    # 基础配置
    assert config.n_rounds == 100
    assert config.random_state is None
    
    # 数据划分配置
    assert config.split_alpha == 0.5
    assert config.test_size == 0.2

def test_custom_activation():
    """测试自定义激活函数"""
    config = FSAConfig(activation=nn.LeakyReLU)
    assert config.activation == nn.LeakyReLU

def test_invalid_mode():
    """测试无效的模式"""
    with pytest.raises(ValueError):
        FSAConfig(mode='invalid')

def test_invalid_censor_rate():
    """测试无效的删失率"""
    with pytest.raises(ValueError):
        FSAConfig(censor_rate=1.5)
    with pytest.raises(ValueError):
        FSAConfig(censor_rate=-0.1)

def test_invalid_client_sample_ratio():
    """测试无效的客户端采样比例"""
    with pytest.raises(ValueError):
        FSAConfig(client_sample_ratio=1.5)
    with pytest.raises(ValueError):
        FSAConfig(client_sample_ratio=0)

def test_invalid_model_type():
    """测试无效的模型类型"""
    with pytest.raises(ValueError):
        FSAConfig(model_type='invalid')

def test_invalid_split_method():
    """测试无效的数据划分方法"""
    with pytest.raises(ValueError):
        FSAConfig(split_method='invalid')

def test_invalid_test_size():
    """测试无效的测试集比例"""
    with pytest.raises(ValueError):
        FSAConfig(test_size=1.5)
    with pytest.raises(ValueError):
        FSAConfig(test_size=0)

def test_invalid_k():
    """测试无效的k参数"""
    with pytest.raises(ValueError):
        FSAConfig(k=0)
    with pytest.raises(ValueError):
        FSAConfig(k=1.1)

def test_invalid_latent_num():
    """测试无效的潜在空间维度"""
    with pytest.raises(ValueError):
        FSAConfig(latent_num=0)
    with pytest.raises(ValueError):
        FSAConfig(latent_num=-1)

def test_invalid_hidden_num():
    """测试无效的隐藏层维度"""
    with pytest.raises(ValueError):
        FSAConfig(hidden_num=0)
    with pytest.raises(ValueError):
        FSAConfig(hidden_num=-1)

def test_invalid_alpha():
    """测试无效的alpha参数"""
    with pytest.raises(ValueError):
        FSAConfig(alpha=-0.1)

def test_invalid_beta():
    """测试无效的beta参数"""
    with pytest.raises(ValueError):
        FSAConfig(beta=-0.1)

def test_invalid_split_alpha():
    """测试无效的split_alpha参数"""
    with pytest.raises(ValueError):
        FSAConfig(split_alpha=0)
    with pytest.raises(ValueError):
        FSAConfig(split_alpha=-0.1)

def test_invalid_num_durations():
    """测试无效的时间离散化数量"""
    with pytest.raises(ValueError):
        FSAConfig(num_durations=0)
    with pytest.raises(ValueError):
        FSAConfig(num_durations=-1)

def test_invalid_dropout():
    """测试无效的dropout率"""
    with pytest.raises(ValueError):
        FSAConfig(dropout=1.1)
    with pytest.raises(ValueError):
        FSAConfig(dropout=-0.1)

def test_invalid_n_rounds():
    """测试无效的轮数"""
    with pytest.raises(ValueError):
        FSAConfig(n_rounds=0)
    with pytest.raises(ValueError):
        FSAConfig(n_rounds=-1)

def test_model_params():
    """测试不同模型类型的默认参数"""
    # PC-Hazard
    config = FSAConfig(model_type='PC-Hazard')
    assert config.model_params == {
        'n_intervals': 10,
        'hidden_size': 32,
        'dropout': 0.1
    }
    
    # LogisticHazard
    config = FSAConfig(model_type='LogisticHazard')
    assert config.model_params == {
        'n_intervals': 10,
        'hidden_size': 32,
        'dropout': 0.1
    }
    
    # DeepHit
    config = FSAConfig(model_type='DeepHit')
    assert config.model_params == {
        'n_intervals': 10,
        'hidden_size': 32,
        'dropout': 0.1
    }
    
    # CoxTime
    config = FSAConfig(model_type='CoxTime')
    assert config.model_params == {
        'hidden_size': 32,
        'dropout': 0.1
    }
    
    # DeepSurv
    config = FSAConfig(model_type='DeepSurv')
    assert config.model_params == {
        'l2_reg': 0.01
    }

    # CoxPH
    config = FSAConfig(model_type='CoxPH', num_nodes=())
    assert config.model_params == {
        'l2_reg': 0.01
    }
    
    # CoxCC
    config = FSAConfig(model_type='CoxCC')
    assert config.model_params == {
        'l2_reg': 0.01
    }

def test_custom_model_params():
    """测试自定义模型参数"""
    custom_params = {
        'n_intervals': 20,
        'hidden_size': 64,
        'dropout': 0.2
    }
    config = FSAConfig(model_type='PC-Hazard', model_params=custom_params)
    assert config.model_params == custom_params

def test_real_mode_requires_dataset_name():
    """测试real模式需要提供数据集名称"""
    with pytest.raises(ValueError):
        FSAConfig(mode='real')

def test_valid_real_mode():
    """测试有效的real模式"""
    config = FSAConfig(mode='real', dataset_name='test_dataset')
    assert config.mode == 'real'
    assert config.dataset_name == 'test_dataset'

if __name__ == '__main__':
    pytest.main([__file__]) 