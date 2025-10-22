import numpy as np
import pandas as pd
import pytest
from federated_survival.data.generator import DataGenerator, SimulationConfig

def test_generator_initialization():
    """测试DataGenerator的初始化"""
    # 使用默认配置
    generator = DataGenerator()
    assert generator.config.n_samples == 200
    assert generator.config.n_features == 15
    assert generator.config.censoring_rate == 0.4
    
    # 使用自定义配置
    config = SimulationConfig(n_samples=100, n_features=10, censoring_rate=0.3, random_state=42)
    generator = DataGenerator(config)
    assert generator.config.n_samples == 100
    assert generator.config.n_features == 10
    assert generator.config.censoring_rate == 0.3

def test_data_structure():
    """测试生成的数据结构"""
    generator = DataGenerator()
    
    # 测试所有支持的数据类型
    for sim_type in ['weibull', 'lognormal', 'SDGM1', 'SDGM2', 'SDGM3', 'SDGM4']:
        data = generator.generate(sim_type)
        
        # 检查返回类型
        assert isinstance(data, pd.DataFrame)
        
        # 检查列名
        expected_columns = [f'x{i+1}' for i in range(generator.config.n_features)] + ['time', 'status']
        assert all(col in data.columns for col in expected_columns)
        
        # 检查数据类型
        assert all(data[f'x{i+1}'].dtype == np.float64 for i in range(generator.config.n_features))
        assert data['time'].dtype == np.float64
        assert data['status'].dtype in [np.int32, np.int64]  # 允许int32或int64
        
        # 检查数据范围
        assert data['status'].isin([0, 1]).all()
        assert (data['time'] >= 0).all()

def test_censoring_rate():
    """测试删失率"""
    generator = DataGenerator()
    
    for sim_type in ['weibull', 'lognormal', 'SDGM1', 'SDGM2', 'SDGM3', 'SDGM4']:
        data = generator.generate(sim_type)
        actual_censoring_rate = 1 - data['status'].mean()
        # 允许一定的误差范围
        assert abs(actual_censoring_rate - generator.config.censoring_rate) < 0.2

def test_random_state():
    """测试随机种子"""
    config = SimulationConfig(random_state=42)
    generator1 = DataGenerator(config)
    generator2 = DataGenerator(config)
    
    for sim_type in ['weibull', 'lognormal', 'SDGM1', 'SDGM2', 'SDGM3', 'SDGM4']:
        data1 = generator1.generate(sim_type)
        data2 = generator2.generate(sim_type)
        
        # 使用相同的随机种子应该生成相同的数据
        pd.testing.assert_frame_equal(data1, data2)

def test_invalid_sim_type():
    """测试无效的模拟类型"""
    generator = DataGenerator()
    
    with pytest.raises(ValueError):
        generator.generate('invalid_type')

def test_data_distribution():
    """测试数据分布"""
    generator = DataGenerator()
    
    # 测试SDGM2的均匀分布特征
    data = generator.generate('SDGM2')
    for i in range(generator.config.n_features):
        assert (data[f'x{i+1}'] >= 0).all() and (data[f'x{i+1}'] <= 1).all()
    
    # 测试SDGM3的伽马分布生存时间
    data = generator.generate('SDGM3')
    assert (data['time'] > 0).all()
    
    # 测试SDGM4的对数正态分布
    data = generator.generate('SDGM4')
    assert (data['time'] > 0).all()

if __name__ == '__main__':
    pytest.main([__file__]) 