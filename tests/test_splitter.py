import pytest
import pandas as pd
import numpy as np
from federated_survival.data.splitter import DataSplitter, DataSet

def test_dataset_structure():
    """测试DataSet结构"""
    # 创建测试数据
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'time': np.random.exponential(scale=1.0, size=n_samples),
        'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 70%删失，30%事件
    })
    
    # 划分数据
    splitter = DataSplitter(n_clients=5, random_state=42)
    result = splitter.split(data)
    
    # 检查返回类型
    assert isinstance(result, DataSet)
    
    # 检查clients_set
    assert isinstance(result.clients_set, dict)
    assert len(result.clients_set) == 5
    for client_id, (X, y) in result.clients_set.items():
        assert client_id.startswith('client')
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == 2  # 两个特征列
        assert y.shape[1] == 2  # time和status两列
        assert np.all(np.isin(y[:, 1], [0, 1]))  # status只能是0或1
    
    # 检查test_data和test_label
    assert isinstance(result.test_data, np.ndarray)
    assert isinstance(result.test_label, np.ndarray)
    assert result.test_data.shape[1] == 2  # 两个特征列
    assert result.test_label.shape[1] == 2  # time和status两列
    assert np.all(np.isin(result.test_label[:, 1], [0, 1]))  # status只能是0或1
    
    # 检查raw_aug_clients_set
    assert isinstance(result.raw_aug_clients_set, dict)
    assert len(result.raw_aug_clients_set) == 0  # 初始为空

def test_iid_split():
    """测试IID划分方式"""
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'time': np.random.exponential(scale=1.0, size=n_samples),
        'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    
    splitter = DataSplitter(n_clients=5, split_type='iid', random_state=42)
    result = splitter.split(data)
    
    # 检查每个客户端的删失比例是否接近原始数据
    original_censoring_rate = 1 - data['status'].mean()
    for client_id, (_, y) in result.clients_set.items():
        censoring_rate = 1 - y[:, 1].mean()
        assert abs(censoring_rate - original_censoring_rate) < 0.1

def test_non_iid_split():
    """测试Non-IID划分方式"""
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'time': np.random.exponential(scale=1.0, size=n_samples),
        'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    
    splitter = DataSplitter(n_clients=5, split_type='non-iid', random_state=42)
    result = splitter.split(data)
    
    # 检查每个客户端的样本数
    total_samples = sum(X.shape[0] for X, _ in result.clients_set.values())
    assert total_samples == int(n_samples * 0.8)  # 80%用于训练

# def test_time_non_iid_split():
#     """测试Time-Non-IID划分方式"""
#     n_samples = 1000
#     data = pd.DataFrame({
#         'x1': np.random.randn(n_samples),
#         'x2': np.random.randn(n_samples),
#         'time': np.random.exponential(scale=1.0, size=n_samples),
#         'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
#     })
    
#     splitter = DataSplitter(n_clients=5, split_type='time-non-iid', random_state=42)
#     result = splitter.split(data)
    
#     # 测试每个客户端的时间分布是否重叠
#     for client_id, (_, y) in result.clients_set.items():
#         # 提取每个客户端的时间分布
#         times = y[:, 0]
#         # 检查时间是否有序
#         assert np.all(times[:-1] <= times[1:])

def test_dirichlet_split():
    """测试Dirichlet划分方式"""
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'time': np.random.exponential(scale=1.0, size=n_samples),
        'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    
    splitter = DataSplitter(n_clients=5, split_type='Dirichlet', alpha=0.5, random_state=42)
    result = splitter.split(data)
    # 输出每个客户端的时间范围
    for client_id, (_, y) in result.clients_set.items():
        times = y[:, 0]
        print(f"Client {client_id}: Time range = [{times.min():.2f}, {times.max():.2f}]")

    # 检查每个客户端的样本数
    total_samples = sum(X.shape[0] for X, _ in result.clients_set.values())

    assert total_samples == int(n_samples * 0.8)  # 80%用于训练


def test_dirichlet_split_repairs_zero_event_clients():
    """Cox-compatible Dirichlet partitions must retain one event per client."""
    rng = np.random.RandomState(3)
    n_samples = 200
    data = pd.DataFrame({
        'x1': rng.randn(n_samples),
        'x2': rng.randn(n_samples),
        'time': rng.exponential(size=n_samples),
        'status': np.r_[np.ones(30), np.zeros(n_samples - 30)],
    })
    result = DataSplitter(
        n_clients=5, split_type='dirichlet', alpha=0.05, random_state=0
    ).split(data)
    assert all(y[:, 1].sum() >= 1 for _, y in result.clients_set.values())
    assert sum(len(x) for x, _ in result.clients_set.values()) == 160


def test_split_rejects_fewer_training_events_than_clients():
    rng = np.random.RandomState(4)
    n_samples = 50
    data = pd.DataFrame({
        'x1': rng.randn(n_samples),
        'x2': rng.randn(n_samples),
        'time': rng.exponential(size=n_samples),
        'status': np.r_[np.ones(5), np.zeros(n_samples - 5)],
    })
    with pytest.raises(ValueError, match='events .* fewer than clients'):
        DataSplitter(
            n_clients=5, split_type='dirichlet', alpha=0.5, random_state=0
        ).split(data)

def test_invalid_split_type():
    """测试无效的划分类型"""
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'time': np.random.exponential(scale=1.0, size=n_samples),
        'status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    
    with pytest.raises(ValueError):
        splitter = DataSplitter(n_clients=5, split_type='invalid')
        splitter.split(data)

if __name__ == '__main__':
    pytest.main([__file__]) 
