"""
客户端类
"""
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
from .differential_privacy import DifferentialPrivacy
from ._losses import apply_cox_patient_normalization
from ._minibatch import deterministic_stream_seed, fit_in_local_steps
from ..models import get_model_adapter
from ..protocols import get_federated_protocol

class Client:
    """客户端类"""
    
    def __init__(self, config, global_model, client_data, client_id, protocol=None):
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
        self.adapter = get_model_adapter(config.model_type)
        self.protocol = protocol or get_federated_protocol(config.federated_protocol)
        if self.protocol.config is None:
            self.protocol.begin_run(config, global_model.state_dict())
        self.local_model = copy.deepcopy(global_model)
        
        # 确保X和y是numpy数组
        self.X = np.array(client_data[0], dtype=np.float32)
        self.y = np.array(client_data[1], dtype=np.float32)
        self.N = len(self.X)
        event_count = float(self.y[:, 1].sum())
        if event_count <= 0:
            raise ValueError(f"Client {client_id} has no observed events")
        self.event_fraction = event_count / self.N
        
        # 转换标签
        self.client_label_transform()
        
        # 初始化差分隐私工具
        if self.config.use_differential_privacy:
            self.dp_tool = DifferentialPrivacy(config)
        else:
            self.dp_tool = None
        
    def client_label_transform(self):
        """标签转换"""
        self.labtrans = getattr(self.config, 'labtrans', None)
        self.y = self.adapter.transform_target(self.y, self.labtrans)
            
    def local_train(self, global_model, epoch: int) -> nn.Module:
        """
        本地训练
        
        Args:
            global_model: 全局模型
            epoch: 当前轮次
            
        Returns:
            nn.Module: 训练后的本地模型
        """
        global_state = {
            name: value.detach().clone()
            for name, value in global_model.state_dict().items()
        }
        self.local_model.load_state_dict(global_state)
            
        self.local_model.train()
        
        # 创建优化器
        optimizer_class = torch.optim.SGD if self.config.optimizer == 'sgd' else torch.optim.Adam
        optimizer = optimizer_class(
            self.local_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        local_model = self.adapter.build_model(
            self.local_model,
            self.config,
            label_transform=self.labtrans,
            optimizer=optimizer,
        )

        apply_cox_patient_normalization(
            local_model, self.config, self.event_fraction
        )
        self.protocol.prepare_client_model(local_model, global_state)
            
        # ``local_epochs`` is retained as a public compatibility name, but its
        # E denotes exactly this many stochastic local optimizer steps per round.
        log = fit_in_local_steps(
            local_model,
            self.X,
            self.y,
            model_type=self.config.model_type,
            batch_size=self.config.batch_size,
            steps=self.config.local_epochs,
            full_batch=self.config.full_batch,
            seed=deterministic_stream_seed(
                self.config.random_seed, str(self.client_id), epoch
            ),
        )
        
        # 如果启用差分隐私，对更新后的权重应用差分隐私保护
        if self.dp_tool is not None:
            private_weights = self.dp_tool.privatize_model_update(
                global_state,
                local_model.net.state_dict(),
                num_clients=self.config.num_clients,
            )
            local_model.net.load_state_dict(private_weights)

        global_vector = torch.cat([
            value.detach().reshape(-1).cpu()
            for value in global_model.parameters()
        ])
        local_vector = torch.cat([
            value.detach().reshape(-1).cpu()
            for value in local_model.net.parameters()
        ])
        delta = local_vector - global_vector
        self.last_drift_norm = float(torch.linalg.vector_norm(delta))
        denominator = self.config.learning_rate * self.config.local_epochs
        self.last_update_direction = -delta / denominator
        self.last_update_direction_norm = float(
            torch.linalg.vector_norm(self.last_update_direction)
        )
        history = log.to_pandas()
        loss_columns = [column for column in history.columns if 'loss' in column]
        self.last_loss = (
            float(history[loss_columns[0]].iloc[-1]) if loss_columns else float('nan')
        )
        
        # 返回训练后的模型
        return local_model.net.eval() 
