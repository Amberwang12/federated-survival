# -*- coding: UTF-8 -*-
"""
示例 04：基础联邦学习 (FSARunner) —— 多种数据划分方式对比

完整演示联邦生存分析的标准流程:
  数据生成 -> 4 种划分方式 (IID / Non-IID / Time-Non-IID / Dirichlet)
  -> FedAvg 训练 -> 评估 -> 对比可视化

方法路径: federated_survival.core.runner.FSARunner.run
          federated_survival.data.splitter.DataSplitter
运行方式: python examples/04_basic_federated.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互后端, 避免弹窗阻塞; 如需弹窗可注释本行
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


# 4 种划分方式配置: (名称, split_type, 额外参数)
PARTITION_TYPES = [
    ("IID",           "iid",          {}),
    ("Non-IID",       "non-iid",      {}),
    ("Time-Non-IID",  "time-non-iid", {}),
    ("Dirichlet",     "Dirichlet",    {"alpha": 0.5}),
]


def describe_clients(dataset):
    """打印各客户端样本数与删失率, 直观体现 non-iid 程度"""
    lines = []
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor_rate = 1.0 - float(np.mean(y[:, 1]))  # status=1 为事件, 0 为删失
        lines.append("    {}: 样本数={:<4d} 删失率={:.1%}".format(
            cid, n, censor_rate))
    return "\n".join(lines)


def run_one_partition(data, split_type, extra_kwargs, num_clients=3,
                      n_features=10, n_samples=300, global_epochs=5):
    """用指定划分方式跑一次完整 FedAvg, 返回 (结果字典, 数据集)"""
    splitter = DataSplitter(
        n_clients=num_clients,
        split_type=split_type,
        test_size=0.2,
        random_state=42,
        **extra_kwargs,
    )
    dataset = splitter.split(data)

    config = FSAConfig(
        num_clients=num_clients,
        n_features=n_features,
        n_samples=n_samples,
        model_type="PC-Hazard",
        local_epochs=1,
        global_epochs=global_epochs,
        learning_rate=0.01,
        batch_size=32,
        random_seed=42,
        client_sample_ratio=1.0,
        early_stopping=False,
        verbose=False,
    )
    runner = FSARunner(config)
    results = runner.run(dataset, type="raw")
    return results, dataset


def main():
    print("=== 示例 04: 基础联邦学习 (多种划分方式对比) ===\n")

    # 1. 生成数据
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    print("数据形状: {}, 总删失率: {:.1%}\n".format(
        data.shape, 1.0 - float(data["status"].mean())))

    # 2. 依次用 4 种划分方式训练并评估
    all_results = {}   # 划分名 -> results
    summary = []       # 汇总行: (名称, 训练C-index, 测试C-index, 训练IBS, 测试IBS)

    for name, split_type, extra in PARTITION_TYPES:
        print("-" * 60)
        print("[{}] 划分方式: {}".format(name, split_type))
        if "alpha" in extra:
            print("  Dirichlet alpha={} (越小越 non-iid)".format(extra["alpha"]))

        results, dataset = run_one_partition(
            data, split_type, extra, global_epochs=5)
        all_results[name] = results

        print("  客户端分布:")
        print(describe_clients(dataset))
        print("  最终训练 C-index: {:.4f} | 测试 C-index: {:.4f}".format(
            results["train_Cindex"][-1], results["test_Cindex"][-1]))
        print("  最终训练 IBS:     {:.4f} | 测试 IBS:     {:.4f}\n".format(
            results["train_IBS"][-1], results["test_IBS"][-1]))

        summary.append((
            name,
            results["train_Cindex"][-1], results["test_Cindex"][-1],
            results["train_IBS"][-1], results["test_IBS"][-1],
        ))

    # 3. 汇总对比表
    print("=" * 60)
    print("汇总对比:")
    print("  {:<14s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "划分方式", "训练C-idx", "测试C-idx", "训练IBS", "测试IBS"))
    for name, tr_c, te_c, tr_b, te_b in summary:
        print("  {:<14s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, tr_c, te_c, tr_b, te_b))

    print("\n指标解读:")
    print("  C-index: 越高越好 (0.5=随机, 1.0=完美), 衡量生存时间排序能力")
    print("  IBS:     越低越好 (0=完美), 衡量生存概率预测准确性")
    print("  划分解读:")
    print("    IID          - 各客户端分布一致, 联邦训练最稳定")
    print("    Non-IID      - 随机划分, 客户端分布可能偏移")
    print("    Time-Non-IID - 按生存时间划分, 模拟时间分布漂移")
    print("    Dirichlet    - alpha 控制异质性, 可构造复杂 non-iid 场景")

    # 4. 可视化: 4 种划分方式的测试 C-index 随轮数变化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, results in all_results.items():
        rounds = range(1, len(results["test_Cindex"]) + 1)
        axes[0].plot(rounds, results["test_Cindex"], marker="o", label=name)
        axes[1].plot(rounds, results["test_IBS"], marker="s", label=name)

    axes[0].set_title("Test C-index vs Round (higher is better)")
    axes[0].set_xlabel("Global Round")
    axes[0].set_ylabel("C-index")
    axes[0].set_xticks(range(1, 6))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test IBS vs Round (lower is better)")
    axes[1].set_xlabel("Global Round")
    axes[1].set_ylabel("IBS")
    axes[1].set_xticks(range(1, 6))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "04_partition_comparison.png")
    plt.savefig(out_path, dpi=120)
    print("\n对比曲线图已保存: {}".format(out_path))

    print("\n=== 示例 04 完成 ===")


if __name__ == "__main__":
    main()
