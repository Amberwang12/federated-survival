# -*- coding: UTF-8 -*-
"""
示例 09：评估指标 (C-index / IBS)

演示生存分析的两个核心评估指标:
  - C-index (一致性指数): 手动调用 calculate_cindex 计算
  - IBS (集成 Brier 分数): 由 FSARunner 内部基于 pycox.EvalSurv 计算

方法路径:
  federated_survival.utils.metrics.calculate_cindex
  FSARunner.run 返回的 train/test_Cindex 与 train/test_IBS
运行方式: python examples/09_evaluation_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from federated_survival.utils.metrics import calculate_cindex
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


def main():
    print("=== 示例 09: 评估指标 ===\n")

    # 1) 手动计算 C-index
    print("1) 手动计算 C-index (calculate_cindex)")
    time = np.array([5.0, 8.0, 12.0, 20.0, 15.0, 10.0])
    event = np.array([1, 1, 0, 1, 1, 0])
    risk_score = np.array([2.1, 1.8, 0.5, 3.0, 2.5, 0.2])
    cindex = calculate_cindex(time, event, risk_score)
    print("   生存时间: {}".format(time.tolist()))
    print("   事件指示: {}".format(event.tolist()))
    print("   风险得分: {}".format(risk_score.tolist()))
    print("   C-index = {:.4f}  (1.0=完美, 0.5=随机)".format(cindex))
    print("   说明: 风险得分越高 -> 预期生存时间越短,")
    print("         calculate_cindex 内部对风险取负号以匹配 concordance_index\n")

    # 2) 联邦训练后的评估指标曲线
    print("2) 联邦训练后的 C-index / IBS (由 FSARunner 内部计算)")
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    dataset = DataSplitter(
        n_clients=3, split_type="iid", test_size=0.2, random_state=42).split(data)

    config = FSAConfig(
        num_clients=3, n_features=10, n_samples=300,
        model_type="PC-Hazard",
        local_epochs=1, global_epochs=3,
        learning_rate=0.01, batch_size=32,
        random_seed=42, verbose=False,
    )
    runner = FSARunner(config)
    res = runner.run(dataset, type="raw")

    print("   训练集 C-index 曲线: {}".format(
        [round(x, 4) for x in res["train_Cindex"]]))
    print("   测试集 C-index 曲线: {}".format(
        [round(x, 4) for x in res["test_Cindex"]]))
    print("   训练集 IBS 曲线:     {}".format(
        [round(x, 4) for x in res["train_IBS"]]))
    print("   测试集 IBS 曲线:     {}".format(
        [round(x, 4) for x in res["test_IBS"]]))

    print("\n指标解读:")
    print("  C-index - 越高越好 (0.5~1.0), 衡量模型对生存时间排序的能力")
    print("  IBS     - 越低越好 (0~0.25), 衡量生存概率预测的准确性")
    print("  训练曲线可观察收敛趋势, 测试曲线反映泛化能力")

    print("\n=== 示例 09 完成 ===")


if __name__ == "__main__":
    main()
