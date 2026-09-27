# -*- coding: UTF-8 -*-
"""
示例 01：数据生成 (DataGenerator)

演示 federated_survival 的模拟数据生成功能，覆盖全部 6 种仿真类型：
  - 加速失效时间 (AFT) 模型: Weibull, Lognormal
  - 比例风险 (PH) 模型:       SDGM1, SDGM4
  - 非比例风险 (non-PH) 模型: SDGM2, SDGM3

方法路径: federated_survival.data.generator.DataGenerator.generate
运行方式: python examples/01_data_generation.py
"""
import os
import sys

# 确保未安装包时也能从项目根目录直接运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from federated_survival.data.generator import DataGenerator, SimulationConfig


def summarize(name, df):
    censor_rate = 1.0 - df["status"].mean()
    print(
        "  [{:<9s}] shape={}, 删失率={:.2%}, "
        "时间范围=[{:.2f}, {:.2f}], 事件数={}".format(
            name, df.shape, censor_rate,
            df["time"].min(), df["time"].max(), int(df["status"].sum()),
        )
    )


def main():
    print("=== 示例 01: 数据生成 ===\n")

    sim_config = SimulationConfig(
        n_samples=500,
        n_features=10,
        random_state=42,
    )
    generator = DataGenerator(sim_config)
    print("配置: n_samples={}, n_features={}".format(
        sim_config.n_samples, sim_config.n_features))
    print("支持的仿真类型: {}\n".format(generator.supported_types))

    # 1) AFT 模型: c_mean 控制删失率 (越大删失越多)
    print("1) Weibull AFT 模型")
    data_weibull = generator.generate("weibull", c_mean=0.4)
    summarize("weibull", data_weibull)

    print("\n2) Lognormal AFT 模型")
    data_lognormal = generator.generate("lognormal", c_mean=0.4)
    summarize("lognormal", data_lognormal)

    # 2) 比例风险模型
    print("\n3) SDGM1 标准比例风险模型")
    data_sdgm1 = generator.generate("SDGM1", c_mean=0.4)
    summarize("SDGM1", data_sdgm1)

    print("\n4) SDGM4 比例风险 + 对数正态误差 (c_step 控制删失)")
    data_sdgm4 = generator.generate("SDGM4", c_step=0.4)
    summarize("SDGM4", data_sdgm4)

    # 3) 非比例风险模型: u_max 控制删失时间上限
    print("\n5) SDGM2 轻度非比例风险 (u_max 控制删失)")
    data_sdgm2 = generator.generate("SDGM2", u_max=4)
    summarize("SDGM2", data_sdgm2)

    print("\n6) SDGM3 强非比例风险 (u_max 控制删失)")
    data_sdgm3 = generator.generate("SDGM3", u_max=7)
    summarize("SDGM3", data_sdgm3)

    # 数据结构
    print("\n数据结构 (以 weibull 为例, 前 3 行):")
    print(data_weibull.head(3).to_string())
    print("\n列名: {}".format(list(data_weibull.columns)))
    print("说明: x1..xp 为特征, time 为观测时间, status 为事件指示(1=发生, 0=删失)")

    print("\n=== 示例 01 完成 ===")


if __name__ == "__main__":
    main()
