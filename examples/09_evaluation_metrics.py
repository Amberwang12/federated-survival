# -*- coding: UTF-8 -*-
"""
Example 09: Evaluation metrics (C-index / IBS)

Demonstrates the two core evaluation metrics for survival analysis:
  - C-index (concordance index): computed manually via calculate_cindex
  - IBS (integrated Brier score): computed inside FSARunner based on pycox.EvalSurv

Method path:
  federated_survival.utils.metrics.calculate_cindex
  train/test_Cindex and train/test_IBS returned by FSARunner.run
Usage: python examples/09_evaluation_metrics.py
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
    print("=== Example 09: Evaluation Metrics ===\n")

    # 1) Compute C-index manually
    print("1) Compute C-index manually (calculate_cindex)")
    time = np.array([5.0, 8.0, 12.0, 20.0, 15.0, 10.0])
    event = np.array([1, 1, 0, 1, 1, 0])
    risk_score = np.array([2.1, 1.8, 0.5, 3.0, 2.5, 0.2])
    cindex = calculate_cindex(time, event, risk_score)
    print("   survival times: {}".format(time.tolist()))
    print("   event indicators: {}".format(event.tolist()))
    print("   risk scores: {}".format(risk_score.tolist()))
    print("   C-index = {:.4f}  (1.0=perfect, 0.5=random)".format(cindex))
    print("   Note: higher risk score -> shorter expected survival time,")
    print("         calculate_cindex negates the risk internally to match concordance_index\n")

    # 2) Evaluation metric curves after federated training
    print("2) C-index / IBS after federated training (computed inside FSARunner)")
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

    print("   train C-index curve: {}".format(
        [round(x, 4) for x in res["train_Cindex"]]))
    print("   test C-index curve:  {}".format(
        [round(x, 4) for x in res["test_Cindex"]]))
    print("   train IBS curve:     {}".format(
        [round(x, 4) for x in res["train_IBS"]]))
    print("   test IBS curve:      {}".format(
        [round(x, 4) for x in res["test_IBS"]]))

    print("\nMetric interpretation:")
    print("  C-index - higher is better (0.5~1.0), measures the model's ability to rank survival times")
    print("  IBS     - lower is better (0~0.25), measures the accuracy of survival probability predictions")
    print("  Training curves show the convergence trend; test curves reflect generalization")

    print("\n=== Example 09 Done ===")


if __name__ == "__main__":
    main()
