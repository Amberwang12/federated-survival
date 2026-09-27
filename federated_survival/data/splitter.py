import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, NamedTuple
from sklearn.model_selection import train_test_split


class DataSet(NamedTuple):
    """数据集，包含clients_set、train_data、train_label、test_data、test_label和raw_aug_clients_set"""
    clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]
    train_data: np.ndarray
    train_label: np.ndarray
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

        if self.n_clients <= 0:
            raise ValueError("n_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.split_type not in [
            'iid', 'random', 'non-iid', 'censoring-non-iid',
            'time-non-iid', 'dirichlet'
        ]:
            raise ValueError(
                "split_type must be one of 'iid', 'random', 'non-iid', "
                "'censoring-non-iid', 'time-non-iid', 'dirichlet'"
            )
        
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
        elif self.split_type == 'random':
            client_data = self._split_random(train_data)
        elif self.split_type in ['non-iid', 'censoring-non-iid']:
            client_data = self._split_censoring_non_iid(train_data)
        elif self.split_type == 'dirichlet':
            # 使用类属性作为默认参数值
            client_data = self._split_Dirichlet(
                train_data, 
                num_of_clients=self.n_clients, 
                beta=self.alpha, 
                n_time_bins=5  # 默认使用5个时间分箱
            )
        else:  # time-non-iid
            client_data = self._split_time_non_iid(train_data)

        # The unified seven-model workflow contains Cox-type objectives whose
        # event sampler is undefined on a client with no observed event.  Keep
        # the requested split as intact as possible by moving one event from a
        # donor only when a generated partition is degenerate.
        client_data = self._ensure_minimum_one_event(client_data)
        
        # 为每个客户端分配数据
        clients_set = {}
        for client_id, client_train_data in client_data.items():
            # 分离特征和标签
            feature_cols = [col for col in client_train_data.columns if col not in ['time', 'status']]
            X = client_train_data[feature_cols].values
            y = client_train_data[['time', 'status']].values
            clients_set[f'client{client_id}'] = (X, y)
        
        # 准备训练数据
        train_X = train_data[feature_cols].values
        train_y = train_data[['time', 'status']].values
        
        # 准备测试数据
        test_X = test_data[feature_cols].values
        test_y = test_data[['time', 'status']].values
        
        # 初始化raw_aug_clients_set为空字典
        raw_aug_clients_set = {}
        
        return DataSet(
            clients_set=clients_set,
            train_data=train_X,
            train_label=train_y,
            test_data=test_X,
            test_label=test_y,
            raw_aug_clients_set=raw_aug_clients_set
        )

    def _ensure_minimum_one_event(
        self, client_data: Dict[int, pd.DataFrame]
    ) -> Dict[int, pd.DataFrame]:
        """Repair a generated partition so every client has one event.

        This is a conditioning constraint for a benchmark that must run Cox
        objectives on every client.  It is not an IID-preserving operation and
        should be reported as such for Dirichlet experiments.
        """
        total_events = int(sum(frame['status'].sum() for frame in client_data.values()))
        if total_events < self.n_clients:
            raise ValueError(
                "Cannot create non-degenerate client partitions: observed "
                f"events ({total_events}) are fewer than clients ({self.n_clients})"
            )

        repaired = {key: value.copy() for key, value in client_data.items()}
        missing = [key for key, frame in repaired.items() if int(frame['status'].sum()) == 0]
        for recipient in missing:
            event_counts = {
                key: int(frame['status'].sum()) for key, frame in repaired.items()
            }
            donor = max(event_counts, key=event_counts.get)
            if event_counts[donor] <= 1:
                raise ValueError("Unable to repair a zero-event client without creating another")
            donor_events = repaired[donor][repaired[donor]['status'] == 1]
            moved = donor_events.iloc[[0]].copy()
            repaired[donor] = repaired[donor].drop(index=moved.index).reset_index(drop=True)
            repaired[recipient] = pd.concat(
                [repaired[recipient], moved], ignore_index=True
            )

        return {key: frame.reset_index(drop=True) for key, frame in repaired.items()}
    
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
    
    def _split_random(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Unstratified random split (still IID in expectation)."""
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

    def _split_censoring_non_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Create a reproducible censoring-rate shift across clients."""
        rng = np.random.RandomState(self.random_state)
        event = data[data['status'] == 1].sample(
            frac=1, random_state=self.random_state
        )
        censored = data[data['status'] == 0].sample(
            frac=1,
            random_state=None if self.random_state is None else self.random_state + 1,
        )
        increasing = np.linspace(1.0, 3.0, self.n_clients)
        event_counts = rng.multinomial(len(event), increasing / increasing.sum())
        censor_counts = rng.multinomial(
            len(censored), increasing[::-1] / increasing.sum()
        )

        def partition(frame, counts):
            pieces, start = [], 0
            for count in counts:
                pieces.append(frame.iloc[start:start + count])
                start += count
            return pieces

        event_parts = partition(event, event_counts)
        censor_parts = partition(censored, censor_counts)
        result = {}
        for client_id in range(self.n_clients):
            client = pd.concat(
                [event_parts[client_id], censor_parts[client_id]], ignore_index=True
            )
            result[client_id] = client.sample(
                frac=1,
                random_state=None if self.random_state is None else self.random_state + client_id,
            )
        return result


    def _split_Dirichlet(self, data: pd.DataFrame, num_of_clients: int, beta: float, n_time_bins: int) -> Dict[int, pd.DataFrame]:
        """
        基于时间分箱 + 事件状态的复合类别狄利克雷 Non-IID 划分。
        将 time 分为 n_time_bins 个区间，与 status (0/1) 组合成 n_time_bins * 2 个伪类别。
        对每个伪类别独立应用 Dirichlet(beta) 分配，实现时间+事件双重异质性。
        
        Args:
            data (pd.DataFrame): 包含 'time' 和 'status' 列的数据框。
            num_of_clients (int): 客户端数量
            beta (float): Dirichlet 分布的 concentration 参数（越小越 Non-IID）
            n_time_bins (int): 时间分箱数量（建议 3~5）
            
        Returns:
            Dict[int, pd.DataFrame]: 键为客户端ID，值为分配给该客户端的子数据框。
        """
        # 1. 对时间列进行分箱
        data = data.copy()
        data['time_bin'] = pd.cut(data['time'], bins=n_time_bins, labels=False)
        
        # 2. 生成复合伪类别：time_bin * 2 + status
        data['pseudo_label'] = data['time_bin'] * 2 + data['status'].astype(int)
        
        # 3. 获取所有唯一的伪类别
        pseudo_labels = data['pseudo_label'].values
        unique_pseudo_labels = np.sort(data['pseudo_label'].unique())
        n_pseudo_labels = len(unique_pseudo_labels)
        
        # 4. 为每个伪类别从狄利克雷分布采样分配比例
        #    shape: (n_pseudo_labels, num_of_clients)
        pseudo_label_proportions = np.random.dirichlet(
            alpha=[beta] * num_of_clients,
            size=n_pseudo_labels
        )

        # 5. 初始化每个客户端的索引列表
        client_indices = [[] for _ in range(num_of_clients)]

        # 6. 遍历每个伪类别，分配其对应的样本
        for pl_idx, pl_value in enumerate(unique_pseudo_labels):
            # 找到属于当前伪类别的所有样本索引
            pl_mask = (pseudo_labels == pl_value)
            pl_sample_indices = np.where(pl_mask)[0]
            n_pl_samples = len(pl_sample_indices)

            # 如果该伪类别没有样本，则跳过
            if n_pl_samples == 0:
                continue

            # 获取该伪类别分配给各客户端的比例
            proportions = pseudo_label_proportions[pl_idx]

            # 使用多项式分布，根据比例将样本数量分配给各个客户端
            # 这确保了分配是离散的且总和等于样本总数
            assigned_counts = np.random.multinomial(n_pl_samples, proportions)

            # 随机打乱当前伪类别的样本索引，以保证随机性
            np.random.shuffle(pl_sample_indices)

            # 根据分配的数量，将样本索引切分并分配给各个客户端
            start = 0
            for client_id, count in enumerate(assigned_counts):
                if count > 0:
                    end = start + count
                    client_indices[client_id].extend(pl_sample_indices[start:end])
                    start = end

        # 7. 根据收集到的索引，构建每个客户端的数据字典
        client_data = {}
        for client_id in range(num_of_clients):
            # 从数据中获取分配给该客户端的样本，并删除临时列
            client_samples = data.iloc[client_indices[client_id]].copy()
            client_samples = client_samples.drop(columns=['time_bin', 'pseudo_label'])
            client_data[client_id] = client_samples.reset_index(drop=True)

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
