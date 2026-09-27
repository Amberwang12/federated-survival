# -*- coding: UTF-8 -*-
"""
示例 10：复现论文 proof 配置

论文配置:
  仿真: SDGM1, n=100
  联邦: 3 clients, Dirichlet alpha=0.8
  模型: DeepSurv (num_nodes=(32,32))
  训练: 5 global rounds, 20 local epochs, batch_size=32
  隐私: Gaussian DP, epsilon=1

为体现差分隐私的影响, 同时跑一组 "无 DP" 对照。

方法路径: federated_survival.core.runner.FSARunner
          federated_survival.core.differential_privacy.DifferentialPrivacy
运行方式: python examples/10_reproduce_paper.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
# pycox 在模型输出溢出时会打 RuntimeWarning, 此处已知是小样本+强DP导致, 抑制以保持输出整洁
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


# ===== 论文 proof 配置 =====
SIM = "SDGM1"
N_SAMPLES = 100
N_FEATURES = 10
N_CLIENTS = 3
SPLIT = "Dirichlet"
ALPHA = 0.8
MODEL = "DeepSurv"
NUM_NODES = (32, 32)
GLOBAL_EPOCHS = 5
LOCAL_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DP_MECHANISM = "gaussian"
DP_EPSILON = 1.0
RANDOM_STATE = 42


def make_config(use_dp):
    """根据论文配置构造 FSAConfig"""
    cfg = dict(
        num_clients=N_CLIENTS,
        n_features=N_FEATURES,
        n_samples=N_SAMPLES,
        model_type=MODEL,
        num_nodes=NUM_NODES,
        local_epochs=LOCAL_EPOCHS,
        global_epochs=GLOBAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_STATE,
        client_sample_ratio=1.0,
        early_stopping=False,
        verbose=False,
    )
    if use_dp:
        cfg.update(
            use_differential_privacy=True,
            dp_mechanism=DP_MECHANISM,
            dp_epsilon=DP_EPSILON,
            dp_delta=1e-5,            # Gaussian 机制必需
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,  # Gaussian 机制必需
            dp_clip_norm=1.0,
        )
    return FSAConfig(**cfg)


def describe_clients(dataset):
    """打印各客户端样本数与删失率"""
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor_rate = 1.0 - float(np.mean(y[:, 1]))
        print("    {}: 样本数={:<4d} 删失率={:.1%}".format(cid, n, censor_rate))


def main():
    print("=" * 64)
    print("示例 10: 复现论文 proof 配置")
    print("=" * 64)

    # 1. 打印配置
    print("\n[配置]")
    print("  仿真:        {} (n={}, d={})".format(SIM, N_SAMPLES, N_FEATURES))
    print("  划分:        {} alpha={}".format(SPLIT, ALPHA))
    print("  客户端数:    {}".format(N_CLIENTS))
    print("  模型:        {} num_nodes={}".format(MODEL, NUM_NODES))
    print("  训练:        global={} local={} batch={}".format(
        GLOBAL_EPOCHS, LOCAL_EPOCHS, BATCH_SIZE))
    print("  隐私:        {} DP, epsilon={}".format(DP_MECHANISM, DP_EPSILON))
    print("  对照:        同时跑一组 无 DP")

    # 2. 生成数据 (SDGM1)
    print("\n[1] 生成 SDGM1 仿真数据...")
    gen = DataGenerator(SimulationConfig(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_STATE))
    data = gen.generate(SIM)
    print("  数据形状: {}, 总删失率: {:.1%}, 事件数: {}".format(
        data.shape, 1.0 - float(data["status"].mean()), int(data["status"].sum())))

    # 3. Dirichlet 划分 (alpha=0.8)
    print("\n[2] Dirichlet 划分 (alpha={})...".format(ALPHA))
    splitter = DataSplitter(
        n_clients=N_CLIENTS, split_type=SPLIT, alpha=ALPHA,
        test_size=0.2, random_state=RANDOM_STATE)
    dataset = splitter.split(data)
    print("  客户端分布:")
    describe_clients(dataset)
    print("  测试集形状: {}".format(dataset.test_data.shape))

    # 4. 打印 DP 噪声规模 (Gaussian 机制)
    print("\n[3] Gaussian DP 噪声参数:")
    dp_cfg = make_config(use_dp=True)
    dp = DifferentialPrivacy(dp_cfg)
    total_eps, per_round_eps = dp.compute_privacy_budget(
        num_rounds=GLOBAL_EPOCHS, num_clients=N_CLIENTS)
    noise_scale = dp.get_noise_scale(num_clients=N_CLIENTS)
    print("    机制          = {}".format(dp_cfg.dp_mechanism))
    print("    epsilon       = {}".format(dp_cfg.dp_epsilon))
    print("    delta         = {}".format(dp_cfg.dp_delta))
    print("    sensitivity   = {}".format(dp_cfg.dp_sensitivity))
    print("    clip_norm     = {}".format(dp_cfg.dp_clip_norm))
    print("    总 eps         = {:.4f}".format(total_eps))
    print("    每轮 eps        = {:.4f}".format(per_round_eps))
    print("    噪声规模 sigma = {:.4f} (考虑 {} 客户端)".format(
        noise_scale, N_CLIENTS))

    # 5. 训练: 有 DP vs 无 DP
    results_all = {}
    for label, use_dp in [("with DP (eps=1)", True), ("without DP", False)]:
        print("\n[4] FedAvg 训练: {} ...".format(label))
        config = make_config(use_dp=use_dp)
        runner = FSARunner(config)
        results = runner.run(dataset, type="raw")
        results_all[label] = results
        print("  训练 C-index 曲线: {}".format(
            ["{:.4f}".format(x) for x in results["train_Cindex"]]))
        print("  测试 C-index 曲线: {}".format(
            ["{:.4f}".format(x) for x in results["test_Cindex"]]))
        print("  测试 IBS     曲线: {}".format(
            ["{:.4f}".format(x) for x in results["test_IBS"]]))
        print("  => 最终测试 C-index={:.4f}, IBS={:.4f}".format(
            results["test_Cindex"][-1], results["test_IBS"][-1]))

    # 6. 汇总对比
    print("\n" + "=" * 64)
    print("汇总对比:")
    print("  {:<20s} {:>12s} {:>12s} {:>12s}".format(
        "实验", "训练C-idx", "测试C-idx", "测试IBS"))
    for label, res in results_all.items():
        print("  {:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            label,
            res["train_Cindex"][-1],
            res["test_Cindex"][-1],
            res["test_IBS"][-1]))

    print("\n解读:")
    print("  DP 通过梯度裁剪 + 噪声注入保护隐私, 通常以轻微性能下降为代价")
    print("  n=100 小样本 + 20 local epochs 易过拟合, DP 噪声反而可能起正则化作用")
    print("  若两组差异很小, 可能因小样本下模型本身拟合能力有限")
    # 针对本次结果的具体诊断
    dp_cindex = results_all["with DP (eps=1)"]["test_Cindex"][-1]
    if np.isnan(dp_cindex) or dp_cindex < 0.3:
        print("\n  [诊断] 本配置下 Gaussian DP (eps=1) 导致模型崩溃:")
        print("    - n=100 极小样本, 每客户端仅约 26 个训练样本")
        print("    - eps=1 噪声规模较大 (sigma~2.8), 梯度信噪比极低")
        print("    - 20 local epochs 使噪声累积, 模型输出溢出 (exp overflow)")
        print("    - 缓解: 增大 eps / 增大 n / 减少 local epochs / 降低 lr")

    # 7. 可视化对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rounds = range(1, GLOBAL_EPOCHS + 1)
    for label, res in results_all.items():
        cindex = np.array(res["test_Cindex"], dtype=float)
        ibs = np.array(res["test_IBS"], dtype=float)
        # C-index: 直接画 (0 值也是有效信息)
        axes[0].plot(rounds, cindex, marker="o", label=label)
        # IBS: nan 用虚线断开
        valid = ~np.isnan(ibs)
        if valid.any():
            axes[1].plot(np.array(list(rounds))[valid], ibs[valid],
                         marker="s", label=label)
        else:
            axes[1].plot([], [], marker="s", label="{} (IBS=nan, model collapsed)".format(label))

    axes[0].set_title("Test C-index vs Round (SDGM1, DeepSurv, Gaussian DP eps=1)")
    axes[0].set_xlabel("Global Round")
    axes[0].set_ylabel("C-index")
    axes[0].set_xticks(list(rounds))
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="random (0.5)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test IBS vs Round")
    axes[1].set_xlabel("Global Round")
    axes[1].set_ylabel("IBS")
    axes[1].set_xticks(list(rounds))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "10_reproduce_paper.png")
    plt.savefig(out_path, dpi=120)
    print("\n对比曲线图已保存: {}".format(out_path))

    print("\n=== 示例 10 完成 ===")


if __name__ == "__main__":
    main()
