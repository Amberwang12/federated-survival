# -*- coding: UTF-8 -*-
"""
示例 05：多模型对比

在同一数据上对比 7 种生存分析模型在联邦学习下的表现:
  PC-Hazard, LogisticHazard, DeepHit, DeepSurv, CoxTime, CoxCC, CoxPH

注意: CoxPH 是线性 Cox 模型 (无隐藏层), 配置时必须令 num_nodes=()。

方法路径: federated_survival.core.config.FSAConfig.model_type
运行方式: python examples/05_model_comparison.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

# 7 种支持的生存模型
MODELS = ["PC-Hazard", "LogisticHazard", "DeepHit",
          "DeepSurv", "CoxTime", "CoxCC", "CoxPH"]


def main():
    print("=== 示例 05: 多模型对比 ===\n")

    # 准备公共数据
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    splitter = DataSplitter(n_clients=3, split_type="iid", test_size=0.2, random_state=42)
    dataset = splitter.split(data)

    summary = []
    for model in MODELS:
        # CoxPH 为线性模型, 不含隐藏层, 必须设 num_nodes=()
        num_nodes = () if model == "CoxPH" else (32, 32)
        config = FSAConfig(
            num_clients=3,
            n_features=10,
            n_samples=300,
            model_type=model,
            num_nodes=num_nodes,
            local_epochs=1,
            global_epochs=3,
            learning_rate=0.01,
            batch_size=32,
            random_seed=42,
            verbose=False,
        )
        runner = FSARunner(config)
        res = runner.run(dataset, type="raw")
        test_cindex = res["test_Cindex"][-1]
        test_ibs = res["test_IBS"][-1]
        summary.append((model, test_cindex, test_ibs))
        print("  {:<16s} 测试 C-index={:.4f}, IBS={:.4f}".format(model, test_cindex, test_ibs))

    # 汇总表
    print("\n=== 汇总 ===")
    print("  {:<16s} {:>10s} {:>10s}".format("模型", "C-index", "IBS"))
    for m, c, i in summary:
        print("  {:<16s} {:>10.4f} {:>10.4f}".format(m, c, i))

    print("\n模型特点:")
    print("  PC-Hazard/LogisticHazard/DeepHit - 离散时间模型, 需时间离散化")
    print("  DeepSurv/CoxPH                   - 比例风险深度/线性模型")
    print("  CoxTime                          - 时间依赖 Cox, 更灵活")
    print("  CoxCC                            - 病例对照 Cox, 适合大数据")
    print("\n=== 示例 05 完成 ===")


if __name__ == "__main__":
    main()
