# -*- coding: UTF-8 -*-
"""
示例 03：数据划分 (DataSplitter)

演示 4 种联邦数据划分方式，并统计各客户端的样本量、删失率与生存时间分布：
  - iid:          各客户端删失率一致(分层), 时间分布相近
  - non-iid:      随机打乱后等分, 各维度均可能轻微偏移
  - time-non-iid: 按生存时间排序后硬切分, 每个客户端只覆盖一个时间区间
  - Dirichlet:    时间分箱+事件状态组成伪类别, 用 Dirichlet(alpha) 采样分配
                  每个客户端覆盖全时间范围, 但各伪类别比例异质 (alpha 越小越 non-iid)

关键区别 (time-non-iid vs Dirichlet):
  - time-non-iid: 时间维度硬切分, 客户端间时间均值单调递增, 不重叠
  - Dirichlet:    时间+事件双重异质, 每个客户端时间范围较广, 但删失率/类别比例差异大
  - 二者区别在时间分布与删失率异质程度, 不只看删失率均值

方法路径: federated_survival.data.splitter.DataSplitter.split
运行方式: python examples/03_data_partitioning.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互后端; 如需弹窗可注释本行
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


def stat_clients(dataset, title):
    """打印各客户端样本数、删失率、时间统计 (mean/median/range)"""
    print("\n[{}]".format(title))
    print("  {:<10s} {:>5s} {:>8s} {:>10s} {:>10s} {:>16s}".format(
        "client", "n", "censor", "t_mean", "t_median", "t_range"))
    times_list = []
    censors = []
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor = 1.0 - y[:, 1].mean()
        t = y[:, 0]
        times_list.append(t)
        censors.append(censor)
        print("  {:<10s} {:>5d} {:>7.1%} {:>10.2f} {:>10.2f}  [{:.1f}, {:.1f}]".format(
            cid, n, censor, t.mean(), np.median(t), t.min(), t.max()))
    # 异质性指标: 删失率标准差 (越大越 non-iid)
    print("  -> 删失率标准差: {:.3f} (越大越 non-iid)".format(np.std(censors)))
    return times_list


def main():
    print("=== 示例 03: 数据划分 ===\n")

    gen = DataGenerator(SimulationConfig(n_samples=600, n_features=10, random_state=42))
    data = gen.generate("SDGM1", c_mean=0.4)
    print("原始数据: {}, 总删失率={:.1%}, 时间范围=[{:.1f}, {:.1f}]\n".format(
        data.shape, 1 - data["status"].mean(), data["time"].min(), data["time"].max()))

    all_times = {}
    for split_type in ["iid", "non-iid", "time-non-iid", "Dirichlet"]:
        splitter = DataSplitter(
            n_clients=4,
            split_type=split_type,
            alpha=0.3,        # Dirichlet 参数, 越小越 non-iid
            test_size=0.2,
            random_state=42,
        )
        dataset = splitter.split(data)
        all_times[split_type] = stat_clients(dataset, split_type)

    # 可视化: 4 种划分下各客户端的时间分布箱线图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (split_type, times_list) in zip(axes, all_times.items()):
        bp = ax.boxplot(times_list, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(split_type)
        ax.set_xlabel("client id")
        ax.set_xticklabels(["c0", "c1", "c2", "c3"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("survival time")
    fig.suptitle("Survival time distribution per client (4 split types)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "03_partition_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print("\n时间分布对比图已保存: {}".format(out_path))

    print("\n说明:")
    print("  IID          - 各客户端删失率一致, 时间分布相近 (理想联邦场景)")
    print("  Non-IID      - 随机划分, 各维度轻微偏移")
    print("  Time-Non-IID - 按时间排序硬切分, 客户端间时间均值单调递增 (c0=短, c3=长)")
    print("                 适合模拟时间分布漂移 (如不同医院收治不同病期患者)")
    print("  Dirichlet    - 时间分箱+事件状态组成伪类别, Dirichlet(alpha) 采样")
    print("                 每个客户端覆盖全时间范围, 但删失率/类别比例异质")
    print("                 alpha 越小越 non-iid (alpha=0.3 时删失率标准差最大)")
    print("\n  关键区别: time-non-iid 是时间硬切分, Dirichlet 是时间+事件双重 soft 异质.")

    print("\n=== 示例 03 完成 ===")


if __name__ == "__main__":
    main()
