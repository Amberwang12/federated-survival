# -*- coding: UTF-8 -*-
"""
Example 05: Multi-Model Comparison

Compare 7 survival analysis models under federated learning on the same data:
  PC-Hazard, LogisticHazard, DeepHit, DeepSurv, CoxTime, CoxCC, CoxPH

Note: CoxPH is a linear Cox model (no hidden layers); you must set num_nodes=().

API path: federated_survival.core.config.FSAConfig.model_type
Run: python examples/05_model_comparison.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner

# 7 supported survival models
MODELS = ["PC-Hazard", "LogisticHazard", "DeepHit",
          "DeepSurv", "CoxTime", "CoxCC", "CoxPH"]


def main():
    print("=== Example 05: Multi-Model Comparison ===\n")

    # Prepare shared data
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    splitter = DataSplitter(n_clients=3, split_type="iid", test_size=0.2, random_state=42)
    dataset = splitter.split(data)

    summary = []
    for model in MODELS:
        # CoxPH is a linear model with no hidden layers; num_nodes must be ()
        num_nodes = () if model == "CoxPH" else (32, 32)
        config = FSAConfig(
            num_clients=3,
            n_features=10,
            n_samples=300,
            model_type=model,
            num_nodes=num_nodes,
            local_epochs=1,
            global_epochs=30,
            learning_rate=0.003,
            batch_size=32,
            random_seed=42,
            verbose=False,
        )
        runner = FSARunner(config)
        res = runner.run(dataset, type="raw")
        test_cindex = res["test_Cindex"][-1]
        test_ibs = res["test_IBS"][-1]
        summary.append((model, test_cindex, test_ibs))
        print("  {:<16s} Test C-index={:.4f}, IBS={:.4f}".format(model, test_cindex, test_ibs))

    # Summary table
    print("\n=== Summary ===")
    print("  {:<16s} {:>10s} {:>10s}".format("Model", "C-index", "IBS"))
    for m, c, i in summary:
        print("  {:<16s} {:>10.4f} {:>10.4f}".format(m, c, i))

    print("\nModel characteristics:")
    print("  PC-Hazard/LogisticHazard/DeepHit - Discrete-time models, require time discretization")
    print("  DeepSurv/CoxPH                   - Deep/linear proportional hazards models")
    print("  CoxTime                          - Time-varying Cox, more flexible")
    print("  CoxCC                            - Case-cohort Cox, suitable for large datasets")
    print("\n=== Example 05 Done ===")


if __name__ == "__main__":
    main()
