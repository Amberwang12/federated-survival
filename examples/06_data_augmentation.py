# -*- coding: UTF-8 -*-
"""
示例 06：数据增强 (MVAEC / MVAES)

演示基于变分自编码器 (VAE) 的两种联邦数据增强方法:
  - MVAEC: 各客户端用本地数据训练 VAE 并生成增强样本 (隐私更强)
  - MVAES: 服务器收集各客户端增强数据后重新分发 (多样性更好)

注意: 增强要求每个客户端至少 10 个样本且至少 1 个未删失样本。

方法路径: FSARunner.run(data, type='raw_aug', aug_method='MVAEC'/'MVAES')
运行方式: python examples/06_data_augmentation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


def main():
    print("=== 示例 06: 数据增强 ===\n")

    # 准备数据 (样本量稍大, 保证各客户端有足够未删失样本训练 VAE)
    gen = DataGenerator(SimulationConfig(n_samples=400, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    splitter = DataSplitter(n_clients=3, split_type="iid", test_size=0.2, random_state=42)
    dataset = splitter.split(data)

    # 公共配置 (含增强参数)
    base = dict(
        num_clients=3, n_features=10, n_samples=400,
        model_type="PC-Hazard",
        local_epochs=1, global_epochs=3,
        learning_rate=0.01, batch_size=32,
        random_seed=42, verbose=False,
        # 数据增强 (VAE) 参数
        latent_num=10,    # 潜在空间维度
        hidden_num=30,    # 隐藏层维度
        alpha=1.0,        # KL 散度权重
        beta=1.0,         # 条件损失权重
        k=0.5,            # 增强比例 (生成样本数为原始的 k 倍)
    )

    # 1) 不使用增强
    print("1) 不使用增强 (raw)")
    r1 = FSARunner(FSAConfig(**base)).run(dataset, type="raw")
    print("   测试 C-index={:.4f}, IBS={:.4f}".format(
        r1["test_Cindex"][-1], r1["test_IBS"][-1]))

    # 2) MVAEC 客户端本地增强
    print("\n2) MVAEC 客户端本地增强")
    r2 = FSARunner(FSAConfig(**base)).run(dataset, type="raw_aug", aug_method="MVAEC")
    print("   测试 C-index={:.4f}, IBS={:.4f}".format(
        r2["test_Cindex"][-1], r2["test_IBS"][-1]))

    # 3) MVAES 服务器集中增强再分发
    print("\n3) MVAES 服务器集中增强")
    r3 = FSARunner(FSAConfig(**base)).run(dataset, type="raw_aug", aug_method="MVAES")
    print("   测试 C-index={:.4f}, IBS={:.4f}".format(
        r3["test_Cindex"][-1], r3["test_IBS"][-1]))

    print("\n选择建议:")
    print("  MVAEC - 隐私优先, 客户端数据不出本地, 通信开销小")
    print("  MVAES - 多样性优先, 服务器再分发, 适合客户端数据量小的场景")
    print("  参数调节: k 控制增强量, latent_num/hidden_num 控制VAE容量")
    print("\n=== 示例 06 完成 ===")


if __name__ == "__main__":
    main()
