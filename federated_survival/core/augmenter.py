import numpy as np
import torch
from typing import Dict, Tuple, Optional
from .mvae import vae_train
import random

class DataAugmenter:
    """数据增强器，支持MVAES和MVAEC两种增强方法"""
    
    def __init__(self, 
                 latent_num: int = 10,
                 hidden_num: int = 30,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        """
        初始化数据增强器
        
        Args:
            latent_num: 潜在空间维度
            hidden_num: 隐藏层维度
            alpha: KL散度权重
            beta: 条件损失权重
        """
        self.latent_num = latent_num
        self.hidden_num = hidden_num
        self.alpha = alpha
        self.beta = beta    
    
    def _aug_client(self, 
                   train_X: np.ndarray,
                   train_y: np.ndarray,
                   k: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个客户端进行数据增强
        
        Args:
            train_X: 特征数据
            train_y: 标签数据
            k: 增强比例
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 增强后的特征数据和标签数据
        """

        # 检查k是否在0-1之间
        if k <= 0 or k > 1:
            raise ValueError("k must be in the range (0, 1]")

        uncensor_num = np.sum(train_y[:, 1] == 1)
        
        # 只使用未删失数据进行训练
        mask = train_y[:, 1] == 1
        vae = vae_train(
            train_X[mask], train_y[mask, 0],
            latent_num=self.latent_num,
            hidden_num=self.hidden_num,
            alpha=self.alpha,
            beta=self.beta
        )
        
        # 采样
        return vae.sample(int(uncensor_num * k))
    
    def _check_clients_set(self, clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """检查客户端数据集是否符合要求"""
        # 检查clients_set是否为空
        if not clients_set:
            raise ValueError("clients_set cannot be empty") 
        
        # 每个客户端数据量不少于10个    
        for i in clients_set.keys():
            if clients_set[i][0].shape[0] < 10:
                raise ValueError("each client must have at least 10 samples")
        
        # 每个客户端必须要有未删失样本
        for i in clients_set.keys():
            if np.sum(clients_set[i][1][:, 1] == 1) == 0:
                raise ValueError("each client must have at least one uncensored sample")

    def mvaes(self, 
            clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]],
            k: float = 1.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        MVAES (Multi-task Variational Autoencoder at the server) 方法
       收集所有客户端增强的数据到服务器，然后分配给每个客户端
        
        Args:
            clients_set: 客户端数据集
            k: 增强比例
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: (增强数据集, 原始+增强数据集)
        """

        self._check_clients_set(clients_set)

        # 收集所有客户端增强的数据到服务器
        all_aug_X, all_aug_y = np.empty(shape=(0, clients_set['client0'][0].shape[1]), dtype=np.float32), np.empty(shape=(0, 2),
                                                                                                 dtype=np.float32)
        self.aug_clients_set = {}
        self.raw_aug_clients_set = {}
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]
            pre = self._aug_client(train_X, train_y, k)
            X_pre, y_pre = pre[0], pre[1]
            self.aug_clients_set[i] = (X_pre, y_pre)
            all_aug_X = np.vstack((all_aug_X, X_pre))
            all_aug_y = np.vstack((all_aug_y, y_pre))
        
        # 分配数据(根据本地非删失的样本数进行增强)

        # 将增强数据分配给每个客户端
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]

            target = int(np.sum(train_y[:, 1] == 1) * k)
            sample_index = random.sample(range(all_aug_X.shape[0]), target) 

            X_pre, y_pre = all_aug_X[sample_index,], all_aug_y[sample_index,]
            raw_aug_X = np.vstack((train_X, X_pre))
            raw_aug_y = np.vstack((train_y, y_pre))
            self.raw_aug_clients_set[i] = (raw_aug_X, raw_aug_y)
        
        return self.raw_aug_clients_set

    def mvaec(self, 
             clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]],
             k: float = 1.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        MVAEC (Multi-task Variational Autoencoder at the client) 方法
        客户端生成的数据对自身进行增强
        
        Args:
            clients_set: 客户端数据集
            k: 增强比例
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: (增强数据集, 原始+增强数据集)
        """

        self._check_clients_set(clients_set)

        self.raw_aug_clients_set = {}
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]

            pre = self._aug_client(train_X, train_y, k)
            X_pre, y_pre = pre[0], pre[1]

            target = int(np.sum(train_y[:, 1] == 1) * k)
            sample_index = random.sample(range(X_pre.shape[0]), target) 

            raw_aug_X = np.vstack((train_X, X_pre[sample_index,]))
            raw_aug_y = np.vstack((train_y, y_pre[sample_index,]))
            self.raw_aug_clients_set[i] = (raw_aug_X, raw_aug_y)

        return self.raw_aug_clients_set

