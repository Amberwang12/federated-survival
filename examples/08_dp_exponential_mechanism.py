# -*- coding: UTF-8 -*-
"""
示例 08：指数机制 (Exponential Mechanism)

差分隐私的指数机制不用于梯度加噪, 而用于离散选择问题
(如模型选择、超参挑选、客户端选择)。它按质量得分以指数概率采样,
高分项更可能被选中, 同时保证隐私。

本示例演示:
  1) 用指数机制从候选模型配置中做私有选择
  2) 对比 Gaussian / Laplace 两种噪声机制对张量的加噪效果

方法路径: federated_survival.core.differential_privacy.DifferentialPrivacy
          .exponential_mechanism / add_gaussian_noise / add_laplace_noise
运行方式: python examples/08_dp_exponential_mechanism.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy


def main():
    print("=== 示例 08: 指数机制 ===\n")

    np.random.seed(42)

    # 指数机制配置
    # 注: epsilon 越大, 分布越陡峭 (高分项越占优); 这里取 10 以便
    # 在有限采样次数内直观看出 "高分高概率" 的趋势。
    config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism="exponential",
        dp_epsilon=10.0,
        dp_sensitivity=1.0,
    )
    dp = DifferentialPrivacy(config)

    # 场景: 5 个候选模型配置, 每个有一个验证集质量得分
    # 得分差距拉开, 便于观察指数机制的概率倾斜
    candidates = torch.randn(5, 20)  # 5 个候选配置, 每个 20 维
    quality_scores = torch.tensor([0.50, 0.65, 0.90, 0.75, 0.80])
    print("候选配置数: {}".format(len(candidates)))
    print("质量得分: {}\n".format(quality_scores.tolist()))

    # 1) 多次采样, 观察选择分布 (高分被选概率更大, 但有随机性)
    print("1) 采样 3000 次的选择分布:")
    counts = np.zeros(5, dtype=int)
    n_trials = 3000
    for _ in range(n_trials):
        idx = dp.exponential_mechanism(candidates, quality_scores)
        counts[idx] += 1
    # 理论概率 P(i) ∝ exp(eps * q_i / (2 * sensitivity)), 用于对照
    scores_np = quality_scores.cpu().numpy()
    theory = np.exp(config.dp_epsilon * scores_np / (2 * config.dp_sensitivity))
    theory = theory / theory.sum()
    print("   候选  得分   选中占比   理论概率")
    for i, (s, c) in enumerate(zip(quality_scores.tolist(), counts)):
        print("     {}  {:.2f}   {:6.1%}    {:6.1%}".format(
            i, s, c / n_trials, theory[i]))

    # 2) 单次选择, 直接返回选中配置的张量
    selected = dp.exponential_mechanism_tensor(candidates, quality_scores)
    print("\n2) 单次选择返回的配置张量形状: {}".format(selected.shape))

    # 3) 噪声机制对比: 对同一张量加 Gaussian / Laplace 噪声
    print("\n3) 噪声机制对比 (对零张量加噪):")
    t = torch.zeros(5)

    config_g = FSAConfig(
        use_differential_privacy=True, dp_mechanism="gaussian",
        dp_epsilon=1.0, dp_delta=1e-5, dp_sensitivity=1.0,
        dp_noise_multiplier=1.0, dp_clip_norm=1.0)
    dp_g = DifferentialPrivacy(config_g)
    g = dp_g.add_gaussian_noise(t)

    config_l = FSAConfig(
        use_differential_privacy=True, dp_mechanism="laplace",
        dp_epsilon=1.0, dp_sensitivity=1.0, dp_clip_norm=1.0)
    dp_l = DifferentialPrivacy(config_l)
    l = dp_l.add_laplace_noise(t)

    print("   原始:       {}".format([round(x, 4) for x in t.tolist()]))
    print("   Gaussian:   {}".format([round(x, 4) for x in g.tolist()]))
    print("   Laplace:    {}".format([round(x, 4) for x in l.tolist()]))

    print("\n说明:")
    print("  指数机制:   离散选择, 按概率采样, 不加噪, 保持输出语义")
    print("  Gaussian:   (eps,delta)-DP, 正态噪声, 对称, 适合深度学习")
    print("  Laplace:    eps-DP, 拉普拉斯噪声, 尾部更重, 适合数值查询")
    print("\n  epsilon 的作用:")
    print("    eps 越大 -> 分布越陡峭, 高分项越占优 (隐私越弱)")
    print("    eps 越小 -> 分布越平坦, 接近均匀采样 (隐私越强)")
    print("    实践中需在 '隐私强度' 与 '选择质量' 之间权衡")
    print("\n=== 示例 08 完成 ===")


if __name__ == "__main__":
    main()
