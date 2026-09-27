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
            feature_columns: 特征列名映射，格式为 {'原始列名': '目标列名'}。
                目标列名可以任意命名，不要求以 'x' 开头；如果为None，
                则除 time/status 外的所有列会被自动重命名为 x1, x2, ...
            time_column: 生存时间列名
            status_column: 事件状态列名
        """
        self.feature_columns = feature_columns
        self.time_column = time_column
        self.status_column = status_column
        # 自动重命名时的 {新列名: 原始列名} 反向映射，用于报错时溯源
        self._auto_renamed = {}
    
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
        
        处理顺序：
            1. 校验 time / status 列存在；
            2. 按 feature_columns 重命名（未提供则自动重命名为 x1, x2, ...）；
            3. 除 time / status 外的所有列均视为特征列，沿用输入数据的列顺序；
            4. 标签列转为 float64 / int32，特征列统一转为 float64；
            5. 输出列顺序为 [特征列..., time, status]。
        
        Args:
            data: 原始数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
            
        Raises:
            ValueError: 缺少 time/status 列；处理后再无特征列；或某个特征列
                无法转换为 float64（例如混入了字符串型的 ID 列）。
        """
        label_columns = (self.time_column, self.status_column)

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
                          if col not in label_columns]
            # 重命名为x1, x2, ...
            new_cols = {col: f'x{i+1}' for i, col in enumerate(feature_cols)}
            # 保留反向映射，报错时能指回原始数据里的列名
            self._auto_renamed = {new: old for old, new in new_cols.items()}
            data = data.rename(columns=new_cols)
        
        # 特征列一律按"排除标签列"确定，并按原始顺序保留。
        # 注意：不可按列名前缀（如 'x'）筛选 —— 否则 feature_columns 映射到
        # 自定义列名时，所有特征会被静默丢弃，只剩标签列。
        feature_cols = [col for col in data.columns if col not in label_columns]
        if not feature_cols:
            raise ValueError(
                "No feature columns found after processing: the data only contains "
                f"the label columns {list(label_columns)}. Check 'feature_columns' "
                "and make sure non-feature columns (ids, dates, ...) are dropped."
            )
        
        # 确保数据类型正确
        data[self.time_column] = data[self.time_column].astype(np.float64)
        data[self.status_column] = data[self.status_column].astype(np.int32)
        
        # 确保特征列是float64类型
        for col in feature_cols:
            try:
                data[col] = data[col].astype(np.float64)
            except (TypeError, ValueError) as exc:
                original = self._auto_renamed.get(col)
                source = f" (from original column '{original}')" if original else ""
                raise ValueError(
                    f"Feature column '{col}'{source} cannot be converted to "
                    f"float64 ({exc}). Drop non-numeric columns such as "
                    "identifiers before loading, or exclude them from "
                    "'feature_columns'."
                ) from exc
        
        # 重新排列列顺序：特征列在前（保持原始顺序），然后是time和status
        final_cols = feature_cols + list(label_columns)
        data = data[final_cols]
        
        return data 