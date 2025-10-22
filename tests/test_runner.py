"""
测试联邦学习分析运行器
"""
import pytest
import torch
import numpy as np
from federated_survival.core.runner import FSARunner
from federated_survival.core.config import FSAConfig
from federated_survival.core.server import Server
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter

@pytest.fixture
def config():
    """创建测试配置"""
    return FSAConfig(
        num_clients=3,
        n_features=10,
        n_samples=100,
        censor_rate=0.3,
        model_type='PC-Hazard',
        local_epochs=2,
        global_epochs=2,
        learning_rate=0.01,
        batch_size=32,
        random_seed=42,
        latent_num=10,
        hidden_num=30,
        alpha=1.0,
        beta=1.0,
        k=0.5
    )

@pytest.fixture
def test_data():
    """生成并划分测试数据"""
    # 生成数据
    sim_config = SimulationConfig(
        n_samples=100,
        n_features=10,
        censoring_rate=0.2,
        random_state=42
    )
    generator = DataGenerator(config=sim_config)
    data = generator.generate('weibull')

    # 划分数据
    splitter = DataSplitter(
        n_clients=3,
        split_type='iid',
        random_state=42
    )
    client_data = splitter.split(data)
    
    return client_data

@pytest.fixture
def mock_data():
    """生成模拟数据"""
    sim_config = SimulationConfig(
        n_samples=100,
        n_features=10,
    )
    generator = DataGenerator(config=sim_config)
    data = generator.generate('weibull')
    return data

def test_runner_initialization(config, test_data):
    """测试运行器初始化"""
    runner = FSARunner(config)
    assert runner.config == config
    assert len(test_data.clients_set) == config.num_clients

def test_set_random_seed(config, test_data):
    """测试随机种子设置"""
    runner = FSARunner(config)
    runner.set_random_seed()
    assert torch.initial_seed() == config.random_seed

def test_get_label_transform(config, test_data):
    """测试标签转换"""
    runner = FSARunner(config)
    labtrans = runner._get_label_transform()
    assert labtrans is not None
    if config.model_type in ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'CoxTime']:
        assert hasattr(labtrans, 'cuts')


def test_get_target(config, test_data):
    """测试目标获取"""
    runner = FSARunner(config)
    # 获取第一个客户端的数据
    client_X, client_y = test_data.clients_set['client0']
    durations, events = runner._get_target(client_y)
    assert client_y.shape[1] == 2
    assert isinstance(client_y, np.ndarray)
    assert isinstance(durations, np.ndarray)
    assert isinstance(events, np.ndarray)
    assert len(durations) == len(events) == len(client_y)
    assert np.all(events >= 0) and np.all(events <= 1)

def test_run_raw_data(config, test_data):
    """测试原始数据运行"""
    runner = FSARunner(config)
    history = runner.run(test_data)
    assert history is not None
    assert 'train_Cindex' in history
    assert 'train_IBS' in history
    assert 'test_Cindex' in history
    assert 'test_IBS' in history
    assert len(history['train_Cindex']) == config.global_epochs
    assert len(history['train_IBS']) == config.global_epochs
    assert len(history['test_Cindex']) == config.global_epochs
    assert len(history['test_IBS']) == config.global_epochs


def test_run_augmented_data(config, test_data):
    """测试使用增强数据运行联邦学习"""
    runner = FSARunner(config)
    # 测试MVAEC增强
    results_mvaec = runner.run(test_data, type='raw_aug', aug_method='MVAEC')
    
    # 验证结果
    assert isinstance(results_mvaec, dict)
    assert 'train_Cindex' in results_mvaec
    assert 'train_IBS' in results_mvaec
    assert 'test_Cindex' in results_mvaec
    assert 'test_IBS' in results_mvaec
    
    # 验证指标范围
    assert all(0 <= cindex <= 1 for cindex in results_mvaec['train_Cindex'])
    assert all(0 <= ibs <= 1 for ibs in results_mvaec['train_IBS'])
    assert all(0 <= cindex <= 1 for cindex in results_mvaec['test_Cindex'])
    assert all(0 <= ibs <= 1 for ibs in results_mvaec['test_IBS'])
    
    # 测试MVAES增强
    results_mvaes = runner.run(test_data, type='raw_aug', aug_method='MVAES')
    
    # 验证结果
    assert isinstance(results_mvaes, dict)
    assert 'train_Cindex' in results_mvaes
    assert 'train_IBS' in results_mvaes
    assert 'test_Cindex' in results_mvaes
    assert 'test_IBS' in results_mvaes
    
    # 验证指标范围
    assert all(0 <= cindex <= 1 for cindex in results_mvaes['train_Cindex'])
    assert all(0 <= ibs <= 1 for ibs in results_mvaes['train_IBS'])
    assert all(0 <= cindex <= 1 for cindex in results_mvaes['test_Cindex'])
    assert all(0 <= ibs <= 1 for ibs in results_mvaes['test_IBS'])
    
    # 测试无效的增强方法
    with pytest.raises(ValueError):
        runner.run(test_data, type='raw_aug', aug_method='invalid_method')
    
    # 测试无效的数据类型
    with pytest.raises(ValueError):
        runner.run(test_data, type='invalid_type', aug_method='MVAEC')

def test_run_with_early_stopping(config, test_data):
    """测试早停运行"""
    config.early_stopping = True
    config.early_stopping_patience = 2
    runner = FSARunner(config)
    history = runner.run(test_data)
    assert history is not None
    assert len(history['train_Cindex']) <= config.global_epochs
    assert len(history['train_IBS']) <= config.global_epochs
    assert len(history['test_Cindex']) <= config.global_epochs
    assert len(history['test_IBS']) <= config.global_epochs

def test_run_with_different_models(config, test_data):
    """测试不同模型运行"""
    model_types = ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'DeepSurv', 'CoxPH', 'CoxTime', 'CoxCC']
    for model_type in model_types:
        config.model_type = model_type
        runner = FSARunner(config)
        history = runner.run(test_data)
        assert history is not None
        assert 'train_Cindex' in history
        assert 'train_IBS' in history
        assert 'test_Cindex' in history
        assert 'test_IBS' in history
        assert len(history['train_Cindex']) == config.global_epochs
        assert len(history['train_IBS']) == config.global_epochs
        assert len(history['test_Cindex']) == config.global_epochs
        assert len(history['test_IBS']) == config.global_epochs

def test_run_with_different_split_methods(config):
    """测试不同划分方法"""
    # 生成数据
    sim_config = SimulationConfig(
        n_samples=100,
        n_features=10,
        censoring_rate=0.3,
        random_state=42
    )
    generator = DataGenerator(config=sim_config)
    data = generator.generate('weibull')

    split_methods = ['iid', 'non-iid', 'time-non-iid']
    for method in split_methods:
        # 划分数据
        splitter = DataSplitter(
            n_clients=3,
            split_type=method,
            random_state=42
        )
        client_data = splitter.split(data)
        
        runner = FSARunner(config)
        history = runner.run(client_data)
        assert history is not None
        assert 'train_Cindex' in history
        assert 'train_IBS' in history
        assert 'test_Cindex' in history
        assert 'test_IBS' in history
        assert len(history['train_Cindex']) == config.global_epochs
        assert len(history['train_IBS']) == config.global_epochs
        assert len(history['test_Cindex']) == config.global_epochs
        assert len(history['test_IBS']) == config.global_epochs 

def test_invalid_model_type():
    """测试无效的模型类型"""
    # 先创建有效的配置
    config = FSAConfig(
        model_type='PC-Hazard',
        num_durations=10,
        n_features=5,
        num_clients=3,
        global_epochs=2,
        local_epochs=1,
        client_sample_ratio=0.5,
        learning_rate=0.01,
        random_seed=42
    )
    runner = FSARunner(config)
    
    # 修改模型类型为无效值
    runner.config.model_type = 'InvalidModel'
    with pytest.raises(ValueError):
        runner._get_label_transform()

def test_invalid_data_type(config, mock_data):
    """测试无效的数据类型"""
    runner = FSARunner(config)
    with pytest.raises(ValueError):
        runner.run(mock_data, type='invalid_type') 