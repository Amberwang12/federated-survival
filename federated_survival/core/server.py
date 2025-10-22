"""
服务器类
"""
import torch
import torch.nn as nn
from typing import Dict, Any
import torchtuples as tt
from pycox.models import (
    PCHazard, LogisticHazard, DeepHitSingle,
    CoxPH, CoxTime, CoxCC
)
from pycox.evaluation import EvalSurv
from pycox.models.cox_time import MLPVanillaCoxTime
import numpy as np
from .differential_privacy import DifferentialPrivacy

class Server:
    """服务器类"""
    
    def __init__(self, config):
        """
        初始化服务器
        
        Args:
            config: 联邦学习配置
        """
        self.config = config
        self.global_model = self._create_model()
        
        # 初始化差分隐私工具
        if self.config.use_differential_privacy:
            self.dp_tool = DifferentialPrivacy(config)
        else:
            self.dp_tool = None
        
    def _create_model(self) -> nn.Module:
        """创建全局模型"""
        # 创建基础网络
        if self.config.model_type == 'CoxTime':
            net = MLPVanillaCoxTime(
                in_features=self.config.n_features,
                num_nodes=self.config.num_nodes,
                batch_norm=self.config.batch_norm,
                dropout=self.config.dropout,
                activation=self.config.activation
            )
        else:
            net = tt.practical.MLPVanilla(
                in_features=self.config.n_features,
                num_nodes=self.config.num_nodes,
                out_features=self.config.out_features,
                batch_norm=self.config.batch_norm,
                dropout=self.config.dropout,
                activation=self.config.activation,
                output_bias=False
            )
        return net
        
    def model_update(self, weight_accumulator: Dict[str, torch.Tensor], num_clients: int = 1):
        """
        更新全局模型
        
        Args:
            weight_accumulator: 权重累加器
            num_clients: 参与聚合的客户端数量
        """
        # 注意:差分隐私噪声在客户端本地训练时已添加,此处不再添加噪声
        # FedAvg算法: 直接用加权平均后的参数替换全局模型参数
        for name, data in self.global_model.state_dict().items():
            data.copy_(weight_accumulator[name])
            
    def model_eval(self, client_set, labtrans) -> tuple:
        """
        评估模型
        计算每个客户端的C-index和IBS, 返回平均值
        Args:
            client_set: 客户端数据
            labtrans: 标签转换器
        Returns:
            tuple: (C-index, IBS)
        """
        model = self._get_model(self.global_model, labtrans)
        c_index = []
        ibs = []
        # 获取训练数据
        for i in client_set.keys():
            test_label = client_set[i][1]
            if self.config.model_type in ['DeepSurv', 'CoxPH', 'CoxTime', 'CoxCC']:
                if labtrans is None:
                    y_train = self._get_target(test_label)
                else:
                    y_train = labtrans.transform(*self._get_target(test_label))
                model.fit(
                    client_set[i][0],
                    y_train,
                    batch_size=2048,
                    epochs=0,
                    verbose=False
                )
                _ = model.compute_baseline_hazards()
            
            surv = model.predict_surv_df(client_set[i][0])
            durations_train, events_train = self._get_target(test_label)
            ev = EvalSurv(surv, durations_train, events_train, censor_surv='km')
            time_grid = np.linspace(durations_train.min(), durations_train.max(), 100)
            c_index.append(ev.concordance_td())
            ibs.append(ev.integrated_brier_score(time_grid))
        return np.mean(c_index), np.mean(ibs)
        
    def _get_model(self, net: nn.Module, labtrans=None):
        """获取生存分析模型"""
        if self.config.model_type == 'PC-Hazard':
            model = PCHazard(net, torch.optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type == 'LogisticHazard':
            model = LogisticHazard(net, torch.optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type == 'DeepHit':
            model = DeepHitSingle(net, torch.optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type in ['DeepSurv', 'CoxPH']:
            model = CoxPH(net, torch.optim.Adam)
        elif self.config.model_type == 'CoxTime':
            model = CoxTime(net, torch.optim.Adam, labtrans=labtrans)
        elif self.config.model_type == 'CoxCC':
            model = CoxCC(net, torch.optim.Adam)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        return model
        
    def _get_target(self, df):
        """获取目标变量"""
        return df[:, 0], df[:, 1] 