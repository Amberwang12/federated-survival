# -*- coding: UTF-8 -*-
"""
示例 02：数据加载 (DataLoader)

演示从 CSV / Excel 文件加载真实生存数据，并自动对齐为框架所需格式
(x1, x2, ..., time, status)。

方法路径: federated_survival.data.loader.DataLoader.load
运行方式: python examples/02_data_loading.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import numpy as np
from federated_survival.data.loader import DataLoader
from federated_survival.data.generator import DataGenerator, SimulationConfig


def main():
    print("=== 示例 02: 数据加载 ===\n")

    # 先用生成器造一份"真实数据"，写成 CSV，列名故意用业务名
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=6, random_state=42))
    df = gen.generate("weibull", c_mean=0.4)
    raw = df.rename(columns={f"x{i+1}": f"feat_{i+1}" for i in range(6)})

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8")
    raw.to_csv(tmp.name, index=False)
    tmp.close()
    print("已生成模拟数据文件: {}".format(tmp.name))
    print("原始列名: {}\n".format(list(raw.columns)))

    # 方式一: 自动重命名特征列 (feature_columns=None)
    print("方式一: 自动重命名特征列为 x1, x2, ...")
    loader_auto = DataLoader(time_column="time", status_column="status")
    data_auto = loader_auto.load(tmp.name)
    print("  加载后列名: {}".format(list(data_auto.columns)))
    print("  形状: {}\n".format(data_auto.shape))

    # 方式二: 显式指定列名映射
    print("方式二: 显式指定列名映射 (feature_columns)")
    mapping = {f"feat_{i+1}": f"x{i+1}" for i in range(6)}
    loader_map = DataLoader(
        feature_columns=mapping, time_column="time", status_column="status")
    data_map = loader_map.load(tmp.name)
    print("  加载后列名: {}".format(list(data_map.columns)))
    print("  前 3 行:")
    print(data_map.head(3).to_string())

    # 支持的格式说明
    print("\n支持的文件格式: .csv / .xlsx / .xls")
    print("要求: 数据必须包含 time 列和 status 列(列名可通过参数自定义)")

    os.unlink(tmp.name)
    print("\n=== 示例 02 完成 ===")


if __name__ == "__main__":
    main()
