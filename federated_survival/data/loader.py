import pandas as pd
import numpy as np
from typing import Union, Dict, Optional
from pathlib import Path

class DataLoader:
    """数据加载器，用于读取Excel和CSV格式的生存数据"""
    
    def __init__(self, 
                 feature_columns: Optional[Dict[str, str]] = None,
                 time_column: str = 'time',
                 status_column: str = 'status'):
        """
        初始化数据加载器
        
        Args:
            feature_columns: 特征列名映射，格式为 {'原始列名': '目标列名'}，如果为None则自动处理
            time_column: 生存时间列名
            status_column: 事件状态列名
        """
        self.feature_columns = feature_columns
        self.time_column = time_column
        self.status_column = status_column
    
    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径，支持.xlsx, .xls, .csv格式
            
        Returns:
            pd.DataFrame: 处理后的数据框，格式与DataGenerator生成的数据一致
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 根据文件扩展名选择读取方法
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        return self._process_data(data)
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据，确保格式与DataGenerator生成的数据一致
        
        Args:
            data: 原始数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        # 确保必要的列存在
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        if self.status_column not in data.columns:
            raise ValueError(f"Status column '{self.status_column}' not found in data")
            
        # 处理特征列
        if self.feature_columns is not None:
            # 重命名特征列
            data = data.rename(columns=self.feature_columns)
        else:
            # 自动处理特征列
            feature_cols = [col for col in data.columns 
                          if col not in [self.time_column, self.status_column]]
            # 重命名为x1, x2, ...
            new_cols = {col: f'x{i+1}' for i, col in enumerate(feature_cols)}
            data = data.rename(columns=new_cols)
        
        # 确保数据类型正确
        data[self.time_column] = data[self.time_column].astype(np.float64)
        data[self.status_column] = data[self.status_column].astype(np.int32)
        
        # 确保特征列是float64类型
        feature_cols = [col for col in data.columns 
                       if col not in [self.time_column, self.status_column]]
        for col in feature_cols:
            data[col] = data[col].astype(np.float64)
        
        # 重新排列列顺序：特征列在前，然后是time和status
        feature_cols = sorted([col for col in data.columns 
                             if col.startswith('x')])
        final_cols = feature_cols + [self.time_column, self.status_column]
        data = data[final_cols]
        
        return data 