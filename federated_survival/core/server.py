"""
服务器类
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from pycox.evaluation import EvalSurv
import numpy as np
from .differential_privacy import DifferentialPrivacy
from ..utils.metrics import evaluation_time_grid
from ..models import get_model_adapter

class Server:
    """服务器类"""
    
    def __init__(self, config):
        """
        初始化服务器
        
        Args:
            config: 联邦学习配置
        """
        self.config = config
        self.adapter = get_model_adapter(config.model_type)
        self.global_model = self._create_model()
        
        # 初始化差分隐私工具
        if self.config.use_differential_privacy:
            self.dp_tool = DifferentialPrivacy(config)
        else:
            self.dp_tool = None
        
    def _create_model(self) -> nn.Module:
        """创建全局模型"""
        return self.adapter.build_network(self.config)
        
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
            data.copy_(weight_accumulator[name].to(dtype=data.dtype, device=data.device))
            
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
        pooled_x = np.concatenate([value[0] for value in client_set.values()], axis=0)
        pooled_y = np.concatenate([value[1] for value in client_set.values()], axis=0)
        self.adapter.prepare_for_prediction(model, pooled_x, pooled_y, labtrans)
        surv, self.last_prediction_diagnostics = self.adapter.predict_survival(
            model,
            pooled_x,
            validation_mode=self.config.prediction_validation,
        )
        durations_train, events_train = self._get_target(pooled_y)
        ev = EvalSurv(surv, durations_train, events_train, censor_surv='km')
        time_grid = evaluation_time_grid(
            durations_train,
            surv.index.values,
            self.config.evaluation_quantiles,
        )
        return float(ev.concordance_td()), float(ev.integrated_brier_score(time_grid))
        
    def _get_model(self, net: nn.Module, labtrans=None):
        """获取生存分析模型"""
        return self.adapter.build_model(
            net,
            self.config,
            label_transform=labtrans,
            optimizer=torch.optim.Adam,
        )
        
    def _get_target(self, df):
        """获取目标变量"""
        return df[:, 0], df[:, 1] 
