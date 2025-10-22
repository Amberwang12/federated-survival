import numpy as np
import pytest
import random
from federated_survival.core.augmenter import DataAugmenter

@pytest.fixture
def sample_data():
    """创建测试用的样本数据"""
    # 创建特征数据
    X = np.random.randn(100, 10).astype(np.float32)
    
    # 创建标签数据 (time, status)
    time = np.random.exponential(scale=1.0, size=100)
    status = np.random.binomial(1, 0.7, size=100)  # 70% 删失
    
    # 创建客户端数据集
    clients_set = {
        'client0': (X[:50], np.column_stack((time[:50], status[:50]))),
        'client1': (X[50:], np.column_stack((time[50:], status[50:])))
    }
    
    return clients_set

@pytest.fixture
def augmenter():
    """创建DataAugmenter实例"""
    return DataAugmenter(
        latent_num=10,
        hidden_num=30,
        alpha=1.0,
        beta=1.0
    )

def test_aug_client(augmenter, sample_data):
    """测试单个客户端增强功能"""
    # 获取第一个客户端的数据
    train_X, train_y = sample_data['client0']
    
    # 测试增强
    X_pre, y_pre = augmenter._aug_client(train_X, train_y, k=0.5)
    
    # 验证结果
    assert isinstance(X_pre, np.ndarray)
    assert isinstance(y_pre, np.ndarray)
    assert X_pre.shape[1] == train_X.shape[1]
    assert y_pre.shape[1] == 2
    assert np.all(np.isfinite(X_pre))
    assert np.all(np.isfinite(y_pre))
    assert np.all(y_pre[:, 1] >= 0)  # status >= 0
    assert np.all(y_pre[:, 1] <= 1)  # status <= 1
    assert np.all(y_pre[:, 0] >= 0)  # time >= 0
    
    # 测试无效的k值
    with pytest.raises(ValueError):
        augmenter._aug_client(train_X, train_y, k=-0.5)
    with pytest.raises(ValueError):
        augmenter._aug_client(train_X, train_y, k=2.0)

def test_mvaes(augmenter, sample_data):
    """测试MVAES增强方法"""
    # 执行MVAES增强
    augmenter.mvaes(sample_data, k=0.5)
    
    # 验证结果结构
    assert isinstance(augmenter.aug_clients_set, dict)
    assert isinstance(augmenter.raw_aug_clients_set, dict)
    assert set(augmenter.aug_clients_set.keys()) == set(sample_data.keys())
    assert set(augmenter.raw_aug_clients_set.keys()) == set(sample_data.keys())
    
    # 验证每个客户端的数据
    for client_id in sample_data.keys():
        original_X, original_y = sample_data[client_id]
        aug_X, aug_y = augmenter.aug_clients_set[client_id]
        raw_aug_X, raw_aug_y = augmenter.raw_aug_clients_set[client_id]
        
        # 验证增强数据集
        assert isinstance(aug_X, np.ndarray)
        assert isinstance(aug_y, np.ndarray)
        assert aug_X.shape[1] == original_X.shape[1]
        assert aug_y.shape[1] == 2
        assert np.all(np.isfinite(aug_X))
        assert np.all(np.isfinite(aug_y))
        
        # 验证原始+增强数据集
        assert isinstance(raw_aug_X, np.ndarray)
        assert isinstance(raw_aug_y, np.ndarray)
        assert raw_aug_X.shape[1] == original_X.shape[1]
        assert raw_aug_y.shape[1] == 2
        assert raw_aug_X.shape[0] > original_X.shape[0]
        
        # 验证原始数据被正确保留
        assert np.array_equal(original_X, raw_aug_X[:original_X.shape[0]])
        assert np.array_equal(original_y, raw_aug_y[:original_y.shape[0]])
        
        # 验证标签的合理性
        assert np.all(raw_aug_y[:, 1] >= 0)
        assert np.all(raw_aug_y[:, 1] <= 1)
        assert np.all(raw_aug_y[:, 0] >= 0)
        
        # 验证增强比例
        target = int(np.sum(original_y[:, 1] == 1) * 0.5)
        assert raw_aug_X.shape[0] - original_X.shape[0] == target

def test_mvaec(augmenter, sample_data):
    """测试MVAEC增强方法"""
    # 执行MVAEC增强
    augmenter.mvaec(sample_data, k=0.5)
    
    # 验证结果结构
    assert isinstance(augmenter.raw_aug_clients_set, dict)
    assert set(augmenter.raw_aug_clients_set.keys()) == set(sample_data.keys())
    
    # 验证每个客户端的数据
    for client_id in sample_data.keys():
        original_X, original_y = sample_data[client_id]
        raw_aug_X, raw_aug_y = augmenter.raw_aug_clients_set[client_id]
        
        # 验证原始+增强数据集
        assert isinstance(raw_aug_X, np.ndarray)
        assert isinstance(raw_aug_y, np.ndarray)
        assert raw_aug_X.shape[1] == original_X.shape[1]
        assert raw_aug_y.shape[1] == 2
        assert raw_aug_X.shape[0] > original_X.shape[0]
        
        # 验证原始数据被正确保留
        assert np.array_equal(original_X, raw_aug_X[:original_X.shape[0]])
        assert np.array_equal(original_y, raw_aug_y[:original_y.shape[0]])
        
        # 验证标签的合理性
        assert np.all(raw_aug_y[:, 1] >= 0)
        assert np.all(raw_aug_y[:, 1] <= 1)
        assert np.all(raw_aug_y[:, 0] >= 0)
        
        # 验证增强比例
        target = int(np.sum(original_y[:, 1] == 1) * 0.5)
        assert raw_aug_X.shape[0] - original_X.shape[0] == target

def test_data_consistency(augmenter, sample_data):
    """测试数据一致性"""
    # 测试MVAES增强
    augmenter.mvaes(sample_data, k=0.5)
    
    # 验证数据一致性
    for client_id in sample_data.keys():
        original_X, original_y = sample_data[client_id]
        raw_aug_X, raw_aug_y = augmenter.raw_aug_clients_set[client_id]
        
        # 验证原始数据被正确保留
        assert np.array_equal(original_X, raw_aug_X[:original_X.shape[0]])
        assert np.array_equal(original_y, raw_aug_y[:original_y.shape[0]])
        
        # 验证增强数据的特征范围
        assert np.all(np.isfinite(raw_aug_X))
        assert np.all(np.isfinite(raw_aug_y))
        
        # 验证标签的合理性
        assert np.all(raw_aug_y[:, 1] >= 0)  # status >= 0
        assert np.all(raw_aug_y[:, 1] <= 1)  # status <= 1
        assert np.all(raw_aug_y[:, 0] >= 0)  # time >= 0
        
        # 验证数据分布
        original_censoring_rate = np.mean(original_y[:, 1] == 0)
        augmented_censoring_rate = np.mean(raw_aug_y[:, 1] == 0)
        assert abs(original_censoring_rate - augmented_censoring_rate) < 0.2  # 允许20%的偏差

def test_edge_cases(augmenter):
    """测试边缘情况"""
    # 测试空数据集
    empty_clients_set = {}
    with pytest.raises(ValueError):
        augmenter.mvaes(empty_clients_set, k=0.5)
    
    # 测试单个样本
    single_sample = {
        'client0': (np.array([[1.0]]), np.array([[1.0, 1.0]]))
    }
    with pytest.raises(ValueError):
        augmenter.mvaes(single_sample, k=0.5)
    
    # 测试没有未删失样本的情况
    no_uncensored = {
        'client0': (np.array([[1.0, 2.0]]), np.array([[1.0, 0.0]]))  # 所有样本都是删失的
    }
    with pytest.raises(ValueError):
        augmenter.mvaes(no_uncensored, k=0.5) 
