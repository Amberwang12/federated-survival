# -*- coding: UTF-8 -*-
"""
示例 07：差分隐私联邦学习 (Gaussian / Laplace)

演示在联邦学习中启用差分隐私保护:
  - Gaussian 机制: (eps, delta)-DP, 适合深度学习梯度, 需 delta 与 noise_multiplier
  - Laplace 机制:  纯 eps-DP, 不需 delta, 适合数值查询

差分隐私通过梯度裁剪 + 噪声注入实现, 仅在客户端本地训练时应用。

方法路径: FSAConfig(use_differential_privacy=True, dp_mechanism=...)
          FSARunner.get_privacy_info()
运行方式: python examples/07_differential_privacy.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


def run_dp(mechanism):
    """以指定 DP 机制运行一次联邦训练, 返回 (测试C-index, 测试IBS)"""
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    dataset = DataSplitter(
        n_clients=3, split_type="iid", test_size=0.2, random_state=42).split(data)

    common = dict(
        num_clients=3, n_features=10, n_samples=300,
        model_type="PC-Hazard",
        local_epochs=1, global_epochs=3,
        learning_rate=0.01, batch_size=32,
        random_seed=42, verbose=False,
        use_differential_privacy=True,
        dp_mechanism=mechanism,
        dp_epsilon=1.0,        # 隐私预算, 越小隐私越强
        dp_sensitivity=1.0,    # 敏感度
        dp_clip_norm=1.0,      # 梯度裁剪范数
    )
    if mechanism == "gaussian":
        common.update(dp_delta=1e-5, dp_noise_multiplier=1.0)

    config = FSAConfig(**common)
    runner = FSARunner(config)

    # 查看隐私信息
    info = runner.get_privacy_info()
    print("  隐私信息: 机制={}, eps={}, 总eps={:.4f}, 每轮eps={:.4f}".format(
        info["mechanism"], info["epsilon"],
        info["total_epsilon"], info["per_round_epsilon"]))
    if mechanism == "gaussian":
        print("           delta={}, 噪声规模={:.4f}, 裁剪范数={}".format(
            info["delta"], info["noise_scale"], info["clip_norm"]))
    else:
        print("           裁剪范数={}".format(info["clip_norm"]))

    res = runner.run(dataset, type="raw")
    return res["test_Cindex"][-1], res["test_IBS"][-1]


def main():
    print("=== 示例 07: 差分隐私联邦学习 ===\n")

    print("1) Gaussian 机制 (eps, delta)-DP")
    c1, i1 = run_dp("gaussian")
    print("   测试 C-index={:.4f}, IBS={:.4f}\n".format(c1, i1))

    print("2) Laplace 机制 eps-DP")
    c2, i2 = run_dp("laplace")
    print("   测试 C-index={:.4f}, IBS={:.4f}\n".format(c2, i2))

    print("机制对比:")
    print("  Gaussian - (eps,delta)-DP, 组合性质好, 适合深度学习多轮训练")
    print("  Laplace  - 纯 eps-DP, 不需 delta, 适合数值/计数查询")
    print("\n隐私-效用权衡:")
    print("  eps 越小 -> 隐私越强, 噪声越大, 模型性能可能下降")
    print("  eps 越大 -> 隐私越弱, 噪声越小, 模型性能更好")
    print("\n=== 示例 07 完成 ===")


if __name__ == "__main__":
    main()
