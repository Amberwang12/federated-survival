# -*- coding: UTF-8 -*-
"""
Example 02: Data Loading (DataLoader)

Demonstrates loading real-world survival data from CSV / Excel files and
automatically aligning it into the format required by the framework
(x1, x2, ..., time, status).

Method path: federated_survival.data.loader.DataLoader.load
Run: python examples/02_data_loading.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import numpy as np
from federated_survival.data.loader import DataLoader
from federated_survival.data.generator import DataGenerator, SimulationConfig


def main():
    print("=== Example 02: Data Loading ===\n")

    # First generate "real data" with the generator, write it to CSV with
    # intentionally business-style column names
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=6, random_state=42))
    df = gen.generate("weibull", c_mean=0.4)
    raw = df.rename(columns={f"x{i+1}": f"feat_{i+1}" for i in range(6)})

    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8")
    raw.to_csv(tmp.name, index=False)
    tmp.close()
    print("Generated simulated data file: {}".format(tmp.name))
    print("Original columns: {}\n".format(list(raw.columns)))

    # Method 1: auto-rename feature columns (feature_columns=None)
    print("Method 1: automatically rename feature columns to x1, x2, ...")
    loader_auto = DataLoader(time_column="time", status_column="status")
    data_auto = loader_auto.load(tmp.name)
    print("  Columns after loading: {}".format(list(data_auto.columns)))
    print("  Shape: {}\n".format(data_auto.shape))

    # Method 2: explicitly specify a column-name mapping
    print("Method 2: explicitly specify a column-name mapping (feature_columns)")
    mapping = {f"feat_{i+1}": f"x{i+1}" for i in range(6)}
    loader_map = DataLoader(
        feature_columns=mapping, time_column="time", status_column="status")
    data_map = loader_map.load(tmp.name)
    print("  Columns after loading: {}".format(list(data_map.columns)))
    print("  First 3 rows:")
    print(data_map.head(3).to_string())

    # Supported formats
    print("\nSupported file formats: .csv / .xlsx / .xls")
    print("Requirement: the data must contain a time column and a status column "
          "(column names can be customized via parameters)")

    os.unlink(tmp.name)
    print("\n=== Example 02 done ===")


if __name__ == "__main__":
    main()
