"""
联邦学习分析运行器
"""
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from pycox.models import (
    PCHazard, LogisticHazard, DeepHitSingle,
    CoxPH, CoxTime, CoxCC
)
from pycox.evaluation import EvalSurv

from .config import FSAConfig
from .server import Server
from .client import Client
from .augmenter import DataAugmenter
from tqdm import tqdm

class FSARunner:
    """联邦学习分析运行器"""
    
    def __init__(self, config: FSAConfig):
        """
        初始化联邦学习分析运行器
        
        Args:
            config: 联邦学习配置
        """
        self.config = config
        self.set_random_seed()
    
    def set_random_seed(self):
        """设置随机种子"""
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_seed)
    
    def _get_label_transform(self):
        """获取标签转换器"""
        if self.config.model_type == 'PC-Hazard':
            labtrans = PCHazard.label_transform(
                self.config.num_durations,
                scheme='quantiles'
            )
        elif self.config.model_type == 'LogisticHazard':
            labtrans = LogisticHazard.label_transform(
                self.config.num_durations,
                scheme='quantiles'
            )
        elif self.config.model_type == 'DeepHit':
            labtrans = DeepHitSingle.label_transform(
                self.config.num_durations,
                scheme='quantiles'
            )
        elif self.config.model_type == 'CoxTime':
            labtrans = CoxTime.label_transform()
        elif self.config.model_type in ['DeepSurv', 'CoxPH', 'CoxCC']:
            labtrans = None
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        return labtrans
    
    def _get_model(self, net: nn.Module, labtrans=None):
        """获取生存分析模型"""
        if self.config.model_type == 'PC-Hazard':
            model = PCHazard(net, optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type == 'LogisticHazard':
            model = LogisticHazard(net, optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type == 'DeepHit':
            model = DeepHitSingle(net, optim.Adam, duration_index=labtrans.cuts)
        elif self.config.model_type in ['DeepSurv', 'CoxPH']:
            model = CoxPH(net, optim.Adam)
        elif self.config.model_type == 'CoxTime':
            model = CoxTime(net, optim.Adam, labtrans=labtrans)
        elif self.config.model_type == 'CoxCC':
            model = CoxCC(net, optim.Adam)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        return model
    
    def run(self, data: Any, type: str = 'raw', aug_method: str = 'MVAEC') -> Dict[str, List[float]]:
        """
        运行联邦学习分析
        
        Args:
            data: 生存数据，需要包含以下属性：
                - clients_set: 客户端数据集，格式为{client_id: (train_X, train_y)}
                - raw_aug_clients_set: 原始数据增强后的客户端数据集，格式为{client_id: (train_X, train_y)}
                - test_data: 测试数据
                - test_label: 测试标签
            type: 数据类型，'raw'或'raw_aug'
            aug_method: 数据增强方法，'MVAEC'或'MVAES'

        Returns:
            dict: {'train_Cindex': train_Cindex, 'train_IBS': train_IBS, 'test_Cindex': test_Cindex, 'test_IBS': test_IBS}
        """
        # 获取标签转换器
        labtrans = self._get_label_transform()
        
        # 获取训练数据
        if type == 'raw':
            clients_set = data.clients_set
        elif type == 'raw_aug':
            augmenter = DataAugmenter(
                latent_num=self.config.latent_num,
                hidden_num=self.config.hidden_num,
                alpha=self.config.alpha,
                beta=self.config.beta
            )
            if aug_method == 'MVAEC':
                clients_set = augmenter.mvaec(data.clients_set, k=self.config.k)
            elif aug_method == 'MVAES':
                clients_set = augmenter.mvaes(data.clients_set, k=self.config.k)
            else:
                raise ValueError(f"Unsupported augmentation method: {aug_method}")
        else:
            raise ValueError(f"Unsupported data type: {type}")
        
        _y_train = np.empty(shape=(0, 2), dtype=np.float32)
        for i in clients_set.keys():
            _y_train = np.vstack((_y_train, clients_set[i][1]))
        y_train = self._get_target(_y_train)

        # 转换标签
        if self.config.model_type in ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'CoxTime']:
            _ = labtrans.fit_transform(*y_train)
            self.config.labtrans = labtrans
            self.config.out_features = labtrans.out_features
        elif self.config.model_type in ['DeepSurv', 'CoxPH', 'CoxCC']:
            self.config.out_features = 1
            
        # 获取测试数据
        durations_test, events_test = self._get_target(data.test_label)
        
        # 初始化服务器和客户端
        server = Server(self.config)
        clients = []
        for i in clients_set.keys():
            clients.append(Client(self.config, server.global_model, clients_set[i], i))
            
        # 如果启用差分隐私，输出隐私保护信息
        if self.config.use_differential_privacy:
            if self.config.verbose:
                mechanism = self.config.dp_mechanism if hasattr(self.config, 'dp_mechanism') else 'gaussian'
                print(f"差分隐私保护已启用:")
                print(f"  - 机制类型: {mechanism.upper()}")
                print(f"  - 隐私预算 (ε): {self.config.dp_epsilon}")
                if mechanism == 'gaussian':
                    print(f"  - 失败概率 (δ): {self.config.dp_delta}")
                    print(f"  - 噪声乘数: {self.config.dp_noise_multiplier}")
                print(f"  - 敏感度: {self.config.dp_sensitivity}")
                if mechanism in ['gaussian', 'laplace']:
                    print(f"  - 梯度裁剪范数: {self.config.dp_clip_norm}")
            
        # 记录指标
        train_Cindex = []
        train_IBS = []
        test_Cindex = []
        test_IBS = []
            
        # 联邦学习训练
        for e in tqdm(range(self.config.global_epochs)):
            # 选择客户端
            candidates = random.sample(
                clients,
                max(round(self.config.client_sample_ratio * self.config.num_clients), 1)
            )
            
            # 初始化权重累加器
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params).to(torch.float)
                
            # 计算总样本数
            N = sum(c.N for c in candidates)
            
            # 客户端训练
            for i, c in enumerate(candidates, 1):
                if self.config.verbose:
                    print(f'Epoch {e+1}/{self.config.global_epochs}, Client {i}/{len(candidates)}')
                    
                local = c.local_train(server.global_model, e)
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(local.state_dict()[name] * c.N / N)
                    
            # 服务器更新
            server.model_update(weight_accumulator, len(candidates))
            
            # 评估模型
            index = server.model_eval(clients_set, labtrans)
            train_Cindex.append(index[0])
            train_IBS.append(index[1])
            
            # 计算测试指标
            model = self._get_model(server.global_model, labtrans)
            
            if self.config.model_type in ['DeepSurv', 'CoxPH', 'CoxTime', 'CoxCC']:
                if labtrans is None:
                    y_train = self._get_target(clients_set['client0'][1])
                else:
                    y_train = labtrans.transform(*self._get_target(clients_set['client0'][1]))
                model.fit(
                    clients_set['client0'][0],
                    y_train,
                    batch_size=2048,
                    epochs=0,
                    verbose=False
                )
                _ = model.compute_baseline_hazards()
                
            surv = model.predict_surv_df(data.test_data)
            ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
            time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
            test_Cindex.append(ev.concordance_td())
            test_IBS.append(ev.integrated_brier_score(time_grid))
            
            if self.config.early_stopping:
                # 提前停止
                if len(train_Cindex) > 5 and all(cindex <= train_Cindex[-6] for cindex in train_Cindex[-5:]):
                    break
                
        return {'train_Cindex': train_Cindex, 'train_IBS': train_IBS, 'test_Cindex': test_Cindex, 'test_IBS': test_IBS}
    
    def get_privacy_info(self) -> Dict[str, Any]:
        """
        获取差分隐私保护信息
        
        Returns:
            dict: 隐私保护相关信息
        """
        if not self.config.use_differential_privacy:
            return {"privacy_protection": False}
        
        from .differential_privacy import DifferentialPrivacy
        dp_tool = DifferentialPrivacy(self.config)
        
        # 计算隐私预算消耗
        total_epsilon, per_round_epsilon = dp_tool.compute_privacy_budget(
            self.config.global_epochs, 
            self.config.num_clients
        )
        
        mechanism = self.config.dp_mechanism if hasattr(self.config, 'dp_mechanism') else 'gaussian'
        
        privacy_info = {
            "privacy_protection": True,
            "mechanism": mechanism,
            "epsilon": self.config.dp_epsilon,
            "sensitivity": self.config.dp_sensitivity,
            "total_epsilon": total_epsilon,
            "per_round_epsilon": per_round_epsilon,
        }
        
        # 添加高斯机制特定参数
        if mechanism == 'gaussian':
            privacy_info.update({
                "delta": self.config.dp_delta,
                "noise_multiplier": self.config.dp_noise_multiplier,
                "noise_scale": dp_tool.get_noise_scale(self.config.num_clients)
            })
        
        # 添加梯度裁剪参数（高斯和拉普拉斯机制）
        if mechanism in ['gaussian', 'laplace']:
            privacy_info["clip_norm"] = self.config.dp_clip_norm
        
        return privacy_info
 
            
    def _get_target(self, df):
        """获取目标变量"""
        return df[:, 0], df[:, 1]
        
    def plot_results(self, results):
        """
        绘制训练结果
        
        Args:
            results: 训练结果，格式为：
                {
                    'train_Cindex': List[float],  # 训练集的C-index
                    'train_IBS': List[float],     # 训练集的IBS
                    'test_Cindex': List[float],   # 测试集的C-index
                    'test_IBS': List[float]       # 测试集的IBS
                }
        """
        # 使用更现代的样式，避免已弃用的seaborn样式
        try:
            # 尝试使用seaborn-v0_8样式（matplotlib 3.6+）
            plt.style.use('seaborn-v0_8')
        except OSError:
            # 如果不可用，使用默认样式并手动设置参数
            plt.style.use('default')
        
        # 设置字体和样式
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # 关闭所有已存在的图形窗口，避免出现多个窗口
        plt.close('all')
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 获取epoch数并转换为列表
        n_epochs = len(results['train_Cindex'])
        epochs = list(range(1, n_epochs + 1))
        
        # 设置C-index的坐标轴范围
        ax1.set_xlim(1, n_epochs)
        cindex_min = min(min(results['train_Cindex']), min(results['test_Cindex']))
        cindex_max = max(max(results['train_Cindex']), max(results['test_Cindex']))
        ax1.set_ylim(cindex_min - 0.05, cindex_max + 0.05)
        
        # 绘制C-index
        ax1.plot(epochs, results['train_Cindex'], 
                color='#2ecc71', 
                label='Training', 
                linewidth=2,
                marker='o',
                markersize=4,
                markevery=5)
        ax1.plot(epochs, results['test_Cindex'], 
                color='#3498db', 
                label='Testing', 
                linewidth=2,
                linestyle='--',
                marker='s',
                markersize=4,
                markevery=5)
        
        ax1.set_title('Concordance Index')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('C-index')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        
        # 添加最终值标注
        final_train_cindex = results['train_Cindex'][-1]
        final_test_cindex = results['test_Cindex'][-1]
        ax1.text(0.02, 0.98, 
                f'Final Train C-index: {final_train_cindex:.3f}\nFinal Test C-index: {final_test_cindex:.3f}',
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 设置IBS的坐标轴范围
        ax2.set_xlim(1, n_epochs)
        ibs_min = min(min(results['train_IBS']), min(results['test_IBS']))
        ibs_max = max(max(results['train_IBS']), max(results['test_IBS']))
        ax2.set_ylim(max(0, ibs_min - 0.05), min(1.0, ibs_max + 0.05))
        
        # 绘制IBS
        ax2.plot(epochs, results['train_IBS'], 
                color='#2ecc71', 
                label='Training', 
                linewidth=2,
                marker='o',
                markersize=4,
                markevery=5)
        ax2.plot(epochs, results['test_IBS'], 
                color='#3498db', 
                label='Testing', 
                linewidth=2,
                linestyle='--',
                marker='s',
                markersize=4,
                markevery=5)
        
        ax2.set_title('Integrated Brier Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IBS')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 添加最终值标注
        final_train_ibs = results['train_IBS'][-1]
        final_test_ibs = results['test_IBS'][-1]
        ax2.text(0.02, 0.98, 
                f'Final Train IBS: {final_train_ibs:.3f}\nFinal Test IBS: {final_test_ibs:.3f}',
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # 调整布局
        plt.tight_layout()
        
        plt.show()
        