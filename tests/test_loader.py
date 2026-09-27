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

def test_feature_columns_without_x_prefix():
    """特征列名不以 x 开头时必须保留（回归：曾因 startswith('x') 被静默丢弃）"""
    data = pd.DataFrame({
        'AGE': [61, 55, 73],
        'BMI': [22.1, 28.4, 24.9],
        'nodes': [3, 0, 7],
        'time': [120, 300, 88],
        'status': [1, 0, 1]
    })
    loader = DataLoader(feature_columns={'AGE': 'age', 'BMI': 'bmi', 'nodes': 'nodes'})
    loaded = loader._process_data(data.copy())

    assert list(loaded.columns) == ['age', 'bmi', 'nodes', 'time', 'status']
    assert loaded.shape == (3, 5)
    assert loaded['bmi'].tolist() == [22.1, 28.4, 24.9]

def test_feature_order_preserved_not_lexicographic():
    """特征数 >= 10 时列序应保持原始顺序，而非 x1, x10, x11, ..., x2"""
    columns = {f'raw{i}': [float(i)] * 3 for i in range(12)}
    columns.update({'time': [1.0, 2.0, 3.0], 'status': [1, 0, 1]})
    data = pd.DataFrame(columns)

    loaded = DataLoader()._process_data(data.copy())

    assert list(loaded.columns) == [f'x{i + 1}' for i in range(12)] + ['time', 'status']

def test_non_numeric_feature_column_raises():
    """混入字符串型 ID 列时应给出指名列名的报错，而非裸 pandas 异常"""
    data = pd.DataFrame({
        'age': [30, 40, 50],
        'pid': ['a', 'b', 'c'],
        'time': [100, 200, 300],
        'status': [1, 0, 1]
    })
    loader = DataLoader()
    # 自动重命名下应同时报出原始列名 pid，而不是只说 x2
    with pytest.raises(ValueError, match="from original column 'pid'"):
        loader._process_data(data.copy())

def test_no_feature_columns_raises():
    """数据只剩标签列时应报错，不能静默返回 0 特征"""
    data = pd.DataFrame({'time': [1.0, 2.0], 'status': [1, 0]})
    with pytest.raises(ValueError, match="No feature columns"):
        DataLoader()._process_data(data.copy())

if __name__ == '__main__':
    pytest.main([__file__]) 