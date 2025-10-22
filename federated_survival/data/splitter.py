import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, NamedTuple
from sklearn.model_selection import train_test_split


class DataSet(NamedTuple):
    """数据集，包含clients_set、test_data、test_label和raw_aug_clients_set"""
    clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]
    test_data: np.ndarray
    test_label: np.ndarray
    raw_aug_clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]


class DataSplitter:
    """数据划分器，支持IID、Non-IID和Time-Non-IID划分方式"""
    
    def __init__(self, 
                 n_clients: int,
                 split_type: str = 'iid',
                 alpha: float = 0.5,
                 test_size: float = 0.2,
                 random_state: Optional[int] = None):
        """
        初始化数据划分器
        
        Args:
            n_clients: 客户端数量
            split_type: 划分类型，可选 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'
            alpha: 狄利克雷分布的参数，用于控制非独立同分布的程度
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.n_clients = n_clients
        self.split_type = split_type.lower()
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state
        
        if self.split_type not in ['iid', 'non-iid', 'time-non-iid', 'dirichlet']:
            raise ValueError("split_type must be one of 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def split(self, data: pd.DataFrame) -> DataSet:
        """
        划分数据
        
        Args:
            data: 输入数据，格式与DataGenerator生成的数据一致
            
        Returns:
            DataSet: 包含clients_set、test_data、test_label和raw_aug_clients_set的数据集
        """
        # 首先划分训练集和测试集，按照删失状态进行分层划分
        train_data, test_data = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=data['status']  # 按照删失状态进行分层
        )

        # 转成float32
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)
        
        # 根据不同的划分方式分配数据
        if self.split_type == 'iid':
            client_data = self._split_iid(train_data)
        elif self.split_type == 'non-iid':
            client_data = self._split_non_iid(train_data)
        elif self.split_type == 'Dirichlet':
            client_data = self._split_Dirichlet(train_data)
        else:  # time-non-iid
            client_data = self._split_time_non_iid(train_data)
        
        # 为每个客户端分配数据
        clients_set = {}
        for client_id, client_train_data in client_data.items():
            # 分离特征和标签
            feature_cols = [col for col in client_train_data.columns if col not in ['time', 'status']]
            X = client_train_data[feature_cols].values
            y = client_train_data[['time', 'status']].values
            clients_set[f'client{client_id}'] = (X, y)
        
        # 准备测试数据
        test_X = test_data[feature_cols].values
        test_y = test_data[['time', 'status']].values
        
        # 初始化raw_aug_clients_set为空字典
        raw_aug_clients_set = {}
        
        return DataSet(
            clients_set=clients_set,
            test_data=test_X,
            test_label=test_y,
            raw_aug_clients_set=raw_aug_clients_set
        )
    
    def _split_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """IID划分方式, 保证每个客户端删失率相同"""
        # 获取特征列
        feature_cols = [col for col in data.columns if col not in ['time', 'status']]
        # 获取删失列
        status_col = [col for col in data.columns if col.startswith('status')][0]
        # 根据删失列进行分层
        data_0 = data[data[status_col] == 0]
        data_1 = data[data[status_col] == 1]
        n_samples_0 = len(data_0)
        n_samples_1 = len(data_1)
        samples_per_client_0 = n_samples_0 // self.n_clients
        samples_per_client_1 = n_samples_1 // self.n_clients
        
        client_data = {}
        for i in range(self.n_clients):
            start_idx_0 = i * samples_per_client_0
            end_idx_0 = (i + 1) * samples_per_client_0 if i < self.n_clients - 1 else n_samples_0
            start_idx_1 = i * samples_per_client_1
            end_idx_1 = (i + 1) * samples_per_client_1 if i < self.n_clients - 1 else n_samples_1
            client_data[i] = pd.concat([data_0.iloc[start_idx_0:end_idx_0].copy(), data_1.iloc[start_idx_1:end_idx_1].copy()])
        
        return client_data
    
    def _split_non_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Non-IID划分方式，不保证每个客户端删失率相同"""
        # 打乱data
        data = data.sample(frac=1).reset_index(drop=True)
        
        n_samples = len(data)
        samples_per_client = n_samples // self.n_clients
        client_data = {}
        for i in range(self.n_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.n_clients - 1 else n_samples
            client_data[i] = data.iloc[start_idx:end_idx].copy()
        return client_data


    def _split_Dirichlet(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Non-IID划分方式，使用狄利克雷分布, 测试中"""
        # 获取特征列
        feature_cols = [col for col in data.columns if col not in ['time', 'status']]
        
        # 对每个特征进行狄利克雷分布采样
        n_features = len(feature_cols)
        proportions = np.random.dirichlet([self.alpha] * self.n_clients, size=n_features)
        
        # 对每个样本分配客户端
        client_indices = [[] for _ in range(self.n_clients)]
        for i, sample in data.iterrows():
            # 计算每个客户端对该样本的权重
            weights = np.ones(self.n_clients)
            for j, col in enumerate(feature_cols):
                feature_value = sample[col]
                # 根据特征值的大小调整权重
                feature_weights = proportions[j] * (1 + np.abs(feature_value))
                weights *= feature_weights
            
            # 归一化权重并选择客户端
            weights = weights / weights.sum()
            client_id = np.random.choice(self.n_clients, p=weights)
            client_indices[client_id].append(i)
        
        # 创建客户端数据
        client_data = {}
        for i in range(self.n_clients):
            client_data[i] = data.loc[client_indices[i]].copy()
        
        return client_data
    
    def _split_time_non_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Time-Non-IID划分方式，基于生存时间进行非独立同分布划分, 区分删失状态"""
        # 获取特征列
        feature_cols = [col for col in data.columns if col not in ['time', 'status']]
        # 获取删失列
        status_col = [col for col in data.columns if col.startswith('status')][0]
        # 根据删失列进行分层
        data_0 = data[data[status_col] == 0]
        data_1 = data[data[status_col] == 1]
        n_samples_0 = len(data_0)
        n_samples_1 = len(data_1)
        
        # 对生存时间进行排序
        sorted_indices_0 = data_0['time'].sort_values().index
        sorted_indices_1 = data_1['time'].sort_values().index
        
        # 将时间分成n_clients个区间
        time_ranges_0 = np.array_split(sorted_indices_0, self.n_clients)
        time_ranges_1 = np.array_split(sorted_indices_1, self.n_clients)
        
        # 对每个时间区间使用狄利克雷分布进行采样
        client_data = {}
        for i in range(self.n_clients):
            # 获取当前时间区间的样本
            time_range_indices_0 = time_ranges_0[i]
            time_range_indices_1 = time_ranges_1[i]
            time_range_data_0 = data_0.loc[time_range_indices_0]
            time_range_data_1 = data_1.loc[time_range_indices_1]

            client_data[i] = pd.concat([time_range_data_0, time_range_data_1])
        
        return client_data 