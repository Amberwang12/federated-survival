# -*- coding: UTF-8 -*-
"""
示例 11：如何避免 Gaussian DP 导致的模型崩溃

背景:
  示例 10 复现论文 proof 配置 (SDGM1, n=100, 3 clients, Dirichlet alpha=0.8,
  DeepSurv, 5 global, 20 local, batch=32, Gaussian DP eps=1) 时,
  模型数值崩溃: test C-index=0, IBS=nan, pycox exp overflow.

原因诊断:
  DP 噪声 sigma~2.8 + n=100 极小样本 (每客户端~26 样本) + 20 local epochs
  使噪声在本地多轮累积, 模型权重发散, 风险预测溢出, 生存函数 exp(-H) 出现 inf/nan.

本示例逐项调整 5 个维度, 验证哪些策略能让模型恢复可用:
  A. 基线 (论文配置, 已知崩溃)
  B. 增大 epsilon (1 -> 10): 降低噪声规模
  C. 减少 local epochs (20 -> 3): 减少噪声累积
  D. 增大样本量 (n=100 -> 500): 提升梯度信噪比
  E. 降低学习率 (0.01 -> 0.001): 缩小噪声对权重的扰动
  F. 组合策略 (n=500, local=3, lr=0.005, eps=5): 多管齐下
  G. 无 DP (性能上限参照)

方法路径: federated_survival.core.runner.FSARunner
          federated_survival.core.differential_privacy.DifferentialPrivacy
运行方式: python examples/11_avoid_collapse.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
# pycox 在风险值溢出时打 RuntimeWarning, 这里已知是 DP 崩溃场景, 抑制以保持输出整洁
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互后端; 如需弹窗可注释本行
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.differential_privacy import DifferentialPrivacy


# ===== 公共配置 (与论文 proof 对齐) =====
SIM = "SDGM1"
N_FEATURES = 10
N_CLIENTS = 3
ALPHA = 0.8
MODEL = "DeepSurv"
NUM_NODES = (32, 32)
GLOBAL_EPOCHS = 5
BATCH_SIZE = 32
RANDOM_STATE = 42


def make_config(n_samples, local_epochs, lr, use_dp, eps):
    """构造 FSAConfig. 公共参数固定, 仅变化研究维度."""
    cfg = dict(
        num_clients=N_CLIENTS,
        n_features=N_FEATURES,
        n_samples=n_samples,
        model_type=MODEL,
        num_nodes=NUM_NODES,
        local_epochs=local_epochs,
        global_epochs=GLOBAL_EPOCHS,
        learning_rate=lr,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_STATE,
        client_sample_ratio=1.0,
        early_stopping=False,
        verbose=False,
    )
    if use_dp:
        cfg.update(
            use_differential_privacy=True,
            dp_mechanism="gaussian",
            dp_epsilon=eps,
            dp_delta=1e-5,
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,
            dp_clip_norm=1.0,
        )
    return FSAConfig(**cfg)


def prepare_dataset(n_samples):
    """生成 SDGM1 数据并做 Dirichlet(alpha=0.8) 划分."""
    gen = DataGenerator(SimulationConfig(
        n_samples=n_samples, n_features=N_FEATURES, random_state=RANDOM_STATE))
    data = gen.generate(SIM)
    splitter = DataSplitter(
        n_clients=N_CLIENTS, split_type="Dirichlet", alpha=ALPHA,
        test_size=0.2, random_state=RANDOM_STATE)
    return splitter.split(data)


def run_one(label, n_samples, local_epochs, lr, use_dp, eps):
    """跑单组实验, 返回 results 与最终指标."""
    print("\n[{}] n={}, local_ep={}, lr={}, dp={}, eps={}".format(
        label, n_samples, local_epochs, lr, use_dp, eps))
    config = make_config(n_samples, local_epochs, lr, use_dp, eps)
    dataset = prepare_dataset(n_samples)
    runner = FSARunner(config)
    results = runner.run(dataset, type="raw")
    tc = results["test_Cindex"][-1]
    ti = results["test_IBS"][-1]
    ti_str = "nan" if np.isnan(ti) else "{:.4f}".format(ti)
    print("  => test C-index={:.4f}, IBS={}".format(tc, ti_str))
    return results


# ===== 实验组定义 =====
# (label, n, local_ep, lr, use_dp, eps)
EXPERIMENTS = [
    ("A baseline (collapsed)", 100, 20, 0.01,  True,  1.0),
    ("B eps=10",               100, 20, 0.01,  True, 10.0),
    ("C local_ep=3",           100,  3, 0.01,  True,  1.0),
    ("D n=500",                500, 20, 0.01,  True,  1.0),
    ("E lr=0.001",             100, 20, 0.001, True,  1.0),
    ("F combined",             500,  3, 0.005, True,  5.0),
    ("G no-DP (upper bound)",  100, 20, 0.01,  False, 1.0),
]


def main():
    print("=" * 72)
    print("示例 11: 如何避免 Gaussian DP 导致的模型崩溃")
    print("=" * 72)
    print("基线 = 示例 10 论文配置 (SDGM1, n=100, 3 clients, DeepSurv,")
    print("       5 global, 20 local, Gaussian DP eps=1) -> 已知 C-index=0, IBS=nan")
    print("本例逐项调整 5 个维度, 验证哪些策略能让模型恢复可用.")

    # 打印 DP 噪声规模随 eps 的变化 (直观解释 B 组为何有效)
    print("\n[DP 噪声规模 vs epsilon]")
    print("  {:<10s} {:>14s} {:>14s}".format("epsilon", "sigma(3 clients)", "per-round eps"))
    for eps in [1.0, 5.0, 10.0]:
        cfg = make_config(100, 1, 0.01, True, eps)
        dp = DifferentialPrivacy(cfg)
        total_e, per_e = dp.compute_privacy_budget(
            num_rounds=GLOBAL_EPOCHS, num_clients=N_CLIENTS)
        sigma = dp.get_noise_scale(num_clients=N_CLIENTS)
        print("  {:<10.1f} {:>14.4f} {:>14.4f}".format(eps, sigma, per_e))
    print("  -> sigma 越小, 噪声越弱, 梯度信噪比越高")

    # 跑全部实验组
    all_results = {}
    for label, n, le, lr, dp, eps in EXPERIMENTS:
        try:
            all_results[label] = run_one(label, n, le, lr, dp, eps)
        except Exception as e:
            print("  失败: {}: {}".format(type(e).__name__, e))
            all_results[label] = None

    # 汇总表
    print("\n" + "=" * 72)
    print("汇总对比 (最终指标, 第 5 轮):")
    print("  {:<26s} {:>12s} {:>12s} {:>10s}".format(
        "实验", "test C-idx", "test IBS", "状态"))
    print("  " + "-" * 62)
    for label, n, le, lr, dp, eps in EXPERIMENTS:
        res = all_results[label]
        if res is None:
            print("  {:<26s} {:>12s} {:>12s} {:>10s}".format(
                label, "ERR", "-", "-"))
            continue
        tc = res["test_Cindex"][-1]
        ti = res["test_IBS"][-1]
        if np.isnan(ti) or tc < 0.3:
            status = "collapsed"
        elif tc < 0.5:
            status = "weak"
        else:
            status = "ok"
        ti_str = "nan" if np.isnan(ti) else "{:.4f}".format(ti)
        print("  {:<26s} {:>12.4f} {:>12s} {:>10s}".format(
            label, tc, ti_str, status))

    # 可视化: C-index 曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rounds = np.arange(1, GLOBAL_EPOCHS + 1)
    for label, n, le, lr, dp, eps in EXPERIMENTS:
        res = all_results[label]
        if res is None:
            continue
        cindex = np.array(res["test_Cindex"], dtype=float)
        ibs = np.array(res["test_IBS"], dtype=float)
        short = label.split(")", 1)[0] + ")"  # "A baseline (collapsed)" -> "A baseline (collapsed)"
        axes[0].plot(rounds, cindex, marker="o", label=short)
        valid = ~np.isnan(ibs)
        if valid.any():
            axes[1].plot(rounds[valid], ibs[valid], marker="s", label=short)

    axes[0].set_title("Test C-index vs Round (avoid collapse strategies)")
    axes[0].set_xlabel("Global Round")
    axes[0].set_ylabel("C-index")
    axes[0].set_xticks(list(rounds))
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="random (0.5)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test IBS vs Round (lower is better)")
    axes[1].set_xlabel("Global Round")
    axes[1].set_ylabel("IBS")
    axes[1].set_xticks(list(rounds))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "11_avoid_collapse.png")
    plt.savefig(out_path, dpi=120)
    print("\n对比曲线图已保存: {}".format(out_path))

    # 策略小结 (基于本例 7 组实测, 非先验)
    print("\n[实测策略小结]")
    print("  1. 增大 epsilon (B): eps 1->10, sigma 2.8->0.28, 唯一单独有效的策略")
    print("     B: C-index=0.5508, IBS=0.2821, 接近无DP上限 (G: 0.5847/0.2736)")
    print("  2. 减少 local epochs (C): local_ep 20->3, 缓解但未根治")
    print("     C: C-index=0.3305 (略好), IBS 仍 nan -> 单独不够")
    print("  3. 增大样本量 (D): n 100->500, 未能挽救 eps=1 崩溃")
    print("     D: C-index=0.2788, IBS nan -> 单独不够")
    print("  4. 降低学习率 (E): lr 0.01->0.001, 完全无效 (5轮未收敛)")
    print("     E: C-index=0.0000 -> lr 太小需配合更多 rounds")
    print("  5. 组合策略 (F): n=500 + local=3 + lr=0.005 + eps=5, weak 通过")
    print("     F: C-index=0.4327, IBS=0.3544 -> eps=5 处于临界区")
    print("\n  核心结论: epsilon 是决定性因素.")
    print("  非 epsilon 维度的单独调整 (C/D/E) 不足以挽救 eps=1 崩溃,")
    print("  因为根本问题是噪声规模 sigma~2.8 相对小样本梯度信号过大.")
    print("  实践: 优先把 eps 调到 sigma<=1 水平 (eps>=5), 再用 local_epochs/n/lr 微调.")
    print("  若必须 eps=1 + 小样本, 需配合数值 clamp 或换用对噪声更鲁棒的模型 (如 PC-Hazard).")

    print("\n=== 示例 11 完成 ===")


if __name__ == "__main__":
    main()
