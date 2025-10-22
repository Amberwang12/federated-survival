import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from federated_survival.data.loader import DataLoader

def test_loader_initialization():
    """测试DataLoader的初始化"""
    # 使用默认配置
    loader = DataLoader()
    assert loader.time_column == 'time'
    assert loader.status_column == 'status'
    assert loader.feature_columns is None
    
    # 使用自定义配置
    feature_cols = {'age': 'x1', 'gender': 'x2'}
    loader = DataLoader(feature_columns=feature_cols,
                       time_column='survival_time',
                       status_column='event')
    assert loader.feature_columns == feature_cols
    assert loader.time_column == 'survival_time'
    assert loader.status_column == 'event'

def test_load_csv():
    """测试加载CSV文件"""
    # 创建测试数据
    data = pd.DataFrame({
        'age': [30, 40, 50],
        'gender': [1, 0, 1],
        'time': [100, 200, 300],
        'status': [1, 0, 1]
    })
    
    # 保存为CSV
    test_file = Path('test_data.csv')
    data.to_csv(test_file, index=False)
    
    try:
        # 加载数据
        loader = DataLoader()
        loaded_data = loader.load(test_file)
        
        # 检查数据结构
        assert isinstance(loaded_data, pd.DataFrame)
        assert set(loaded_data.columns) == {'x1', 'x2', 'time', 'status'}
        assert loaded_data['time'].dtype == np.float64
        assert loaded_data['status'].dtype == np.int32
        assert loaded_data['x1'].dtype == np.float64
        assert loaded_data['x2'].dtype == np.float64
        
        # 检查数据值
        assert (loaded_data['time'] == [100, 200, 300]).all()
        assert (loaded_data['status'] == [1, 0, 1]).all()
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()

def test_load_excel():
    """测试加载Excel文件"""
    # 创建测试数据
    data = pd.DataFrame({
        'age': [30, 40, 50],
        'gender': [1, 0, 1],
        'time': [100, 200, 300],
        'status': [1, 0, 1]
    })
    
    # 保存为Excel
    test_file = Path('test_data.xlsx')
    data.to_excel(test_file, index=False)
    
    try:
        # 加载数据
        loader = DataLoader()
        loaded_data = loader.load(test_file)
        
        # 检查数据结构
        assert isinstance(loaded_data, pd.DataFrame)
        assert set(loaded_data.columns) == {'x1', 'x2', 'time', 'status'}
        assert loaded_data['time'].dtype == np.float64
        assert loaded_data['status'].dtype == np.int32
        assert loaded_data['x1'].dtype == np.float64
        assert loaded_data['x2'].dtype == np.float64
        
        # 检查数据值
        assert (loaded_data['time'] == [100, 200, 300]).all()
        assert (loaded_data['status'] == [1, 0, 1]).all()
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()

def test_custom_feature_columns():
    """测试自定义特征列名"""
    # 创建测试数据
    data = pd.DataFrame({
        'age': [30, 40, 50],
        'gender': [1, 0, 1],
        'time': [100, 200, 300],
        'status': [1, 0, 1]
    })
    
    # 保存为CSV
    test_file = Path('test_data.csv')
    data.to_csv(test_file, index=False)
    
    try:
        # 使用自定义特征列名
        feature_cols = {'age': 'x1', 'gender': 'x2'}
        loader = DataLoader(feature_columns=feature_cols)
        loaded_data = loader.load(test_file)
        
        # 检查列名
        assert set(loaded_data.columns) == {'x1', 'x2', 'time', 'status'}
        
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()

def test_invalid_file():
    """测试无效文件"""
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load('nonexistent_file.csv')

def test_invalid_format():
    """测试无效文件格式"""
    # 创建测试文件
    test_file = Path('test_data.txt')
    test_file.write_text('test')
    
    try:
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load(test_file)
    finally:
        if test_file.exists():
            test_file.unlink()

def test_missing_columns():
    """测试缺少必要列"""
    # 创建测试数据（缺少time列）
    data = pd.DataFrame({
        'age': [30, 40, 50],
        'gender': [1, 0, 1],
        'status': [1, 0, 1]
    })
    
    # 保存为CSV
    test_file = Path('test_data.csv')
    data.to_csv(test_file, index=False)
    
    try:
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load(test_file)
    finally:
        if test_file.exists():
            test_file.unlink()

if __name__ == '__main__':
    pytest.main([__file__]) 