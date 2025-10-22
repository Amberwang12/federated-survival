"""
客户端类
"""
import copy
import torch
import torch.nn as nn
import torchtuples as tt
import numpy as np
from typing import Tuple, Dict, Any
from pycox.models import (
    PCHazard, LogisticHazard, DeepHitSingle,
    CoxPH, CoxTime, CoxCC
)
from .differential_privacy import DifferentialPrivacy

class Client:
    """客户端类"""
    
    def __init__(self, config, global_model, client_data, client_id):
        """
        初始化客户端
        
        Args:
            config: 联邦学习配置
            global_model: 全局模型
            client_data: 客户端数据
            client_id: 客户端ID
        """
        self.config = config
        self.client_id = client_id
        self.local_model = copy.deepcopy(global_model)
        
        # 确保X和y是numpy数组
        self.X = np.array(client_data[0], dtype=np.float32)
        self.y = np.array(client_data[1], dtype=np.float32)
        self.N = len(self.X)
        
        # 转换标签
        self.client_label_transform()
        
        # 初始化差分隐私工具
        if self.config.use_differential_privacy:
            self.dp_tool = DifferentialPrivacy(config)
        else:
            self.dp_tool = None
        
    def client_label_transform(self):
        """标签转换"""
        get_target = lambda df: (df[:, 0], df[:, 1])
        if self.config.model_type in ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'CoxTime']:
            self.labtrans = self.config.labtrans
            self.y = self.labtrans.transform(*get_target(self.y))
        else:
            self.y = get_target(self.y)
            
    def local_train(self, global_model, epoch: int) -> nn.Module:
        """
        本地训练
        
        Args:
            global_model: 全局模型
            epoch: 当前轮次
            
        Returns:
            nn.Module: 训练后的本地模型
        """
        # 更新本地模型
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
            
        self.local_model.train()
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            self.local_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.05
        )
        
        # 创建模型
        if self.config.model_type == 'PC-Hazard':
            local_model = PCHazard(self.local_model, optimizer, duration_index=self.labtrans.cuts)
        elif self.config.model_type == 'LogisticHazard':
            local_model = LogisticHazard(self.local_model, optimizer, duration_index=self.labtrans.cuts)
        elif self.config.model_type == 'DeepHit':
            local_model = DeepHitSingle(self.local_model, optimizer, duration_index=self.labtrans.cuts)
        elif self.config.model_type in ['DeepSurv', 'CoxPH']:
            local_model = CoxPH(self.local_model, optimizer)
        elif self.config.model_type == 'CoxTime':
            local_model = CoxTime(self.local_model, optimizer, labtrans=self.labtrans)
        elif self.config.model_type == 'CoxCC':
            local_model = CoxCC(self.local_model, optimizer)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
        # 训练模型
        if self.config.model_type == 'PC-Hazard':
            local_model.fit(
                self.X, self.y,
                epochs=self.config.local_epochs,
                batch_size=self.N,
                verbose=False,
                check_out_features=False
            )
        else:
            local_model.fit(
                self.X, self.y,
                epochs=self.config.local_epochs,
                batch_size=self.N,
                verbose=False
            )
        
        # 如果启用差分隐私，对梯度应用差分隐私保护
        if self.dp_tool is not None:
            # 保存训练后的模型权重快照（应用DP前）
            weights_before_dp = {name: param.clone().detach() 
                                for name, param in local_model.net.state_dict().items()}
            # print(weights_before_dp)
            
            # 应用差分隐私，使用配置中指定的机制
            mechanism = self.config.dp_mechanism if hasattr(self.config, 'dp_mechanism') else 'gaussian'
            self.dp_tool.apply_dp_to_gradients(local_model.net, optimizer, mechanism=mechanism)
            
            # # 检查权重是否发生变化（应用DP后）
            # weights_after_dp = local_model.net.state_dict()
            
            # # 计算权重差异
            # total_diff = 0.0
            # param_count = 0
            # for name in weights_before_dp.keys():
            #     diff = torch.sum(torch.abs(weights_after_dp[name] - weights_before_dp[name])).item()
            #     total_diff += diff
            #     param_count += weights_after_dp[name].numel()
            
            # avg_diff = total_diff / param_count if param_count > 0 else 0.0
            
            # if epoch == 0:  # 只在第一轮打印
            #     print(f"\n[Client {self.client_id}] 差分隐私检查:")
            #     print(f"  训练后应用DP前后权重平均差异: {avg_diff:.10f}")
            #     print(f"  总参数数量: {param_count}")
            #     if avg_diff == 0.0:
            #         print(f"  ⚠️ 警告: 权重没有变化，差分隐私可能未生效！")
            #     else:
            #         print(f"  ✓ 权重已改变，DP噪声已添加")
            
        # 返回训练后的模型
        return local_model.net.eval() 