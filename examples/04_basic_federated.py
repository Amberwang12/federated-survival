# -*- coding: UTF-8 -*-
"""
Example 04: Basic Federated Learning (FSARunner) — Comparison of Partitioning Schemes

End-to-end demonstration of the standard federated survival analysis workflow:
  Data generation -> 4 partitioning schemes (IID / Non-IID / Time-Non-IID / Dirichlet)
  -> FedAvg training -> evaluation -> comparison visualization

Method path: federated_survival.core.runner.FSARunner.run
          federated_survival.data.splitter.DataSplitter
Run: python examples/04_basic_federated.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend to avoid blocking windows; comment out to show windows
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


# Configuration for 4 partitioning schemes: (name, split_type, extra kwargs)
PARTITION_TYPES = [
    ("IID",           "iid",          {}),
    ("Non-IID",       "non-iid",      {}),
    ("Time-Non-IID",  "time-non-iid", {}),
    ("Dirichlet",     "Dirichlet",    {"alpha": 0.5}),
]


def describe_clients(dataset):
    """Print per-client sample size and censoring rate to show the degree of non-iidness"""
    lines = []
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor_rate = 1.0 - float(np.mean(y[:, 1]))  # status=1 is event, 0 is censored
        lines.append("    {}: n={:<4d} censoring rate={:.1%}".format(
            cid, n, censor_rate))
    return "\n".join(lines)


def run_one_partition(data, split_type, extra_kwargs, num_clients=3,
                      n_features=10, n_samples=300, global_epochs=30):
    """Run a full FedAvg pass with the given partitioning scheme; return (results dict, dataset)"""
    splitter = DataSplitter(
        n_clients=num_clients,
        split_type=split_type,
        test_size=0.2,
        random_state=42,
        **extra_kwargs,
    )
    dataset = splitter.split(data)

    config = FSAConfig(
        num_clients=num_clients,
        n_features=n_features,
        n_samples=n_samples,
        model_type="PC-Hazard",
        local_epochs=1,
        global_epochs=global_epochs,
        learning_rate=0.003,
        batch_size=32,
        random_seed=42,
        client_sample_ratio=1.0,
        early_stopping=False,
        verbose=False,
    )
    runner = FSARunner(config)
    results = runner.run(dataset, type="raw")
    return results, dataset


def main():
    print("=== Example 04: Basic Federated Learning (partitioning scheme comparison) ===\n")

    # 1. Generate data
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    print("Data shape: {}, overall censoring rate: {:.1%}\n".format(
        data.shape, 1.0 - float(data["status"].mean())))

    # 2. Train and evaluate sequentially with the 4 partitioning schemes
    all_results = {}   # partition name -> results
    summary = []       # summary rows: (name, train C-index, test C-index, train IBS, test IBS)

    for name, split_type, extra in PARTITION_TYPES:
        print("-" * 60)
        print("[{}] partitioning scheme: {}".format(name, split_type))
        if "alpha" in extra:
            print("  Dirichlet alpha={} (smaller = more non-iid)".format(extra["alpha"]))

        results, dataset = run_one_partition(
            data, split_type, extra, global_epochs=5)
        all_results[name] = results

        print("  Client distribution:")
        print(describe_clients(dataset))
        print("  Final train C-index: {:.4f} | test C-index: {:.4f}".format(
            results["train_Cindex"][-1], results["test_Cindex"][-1]))
        print("  Final train IBS:     {:.4f} | test IBS:     {:.4f}\n".format(
            results["train_IBS"][-1], results["test_IBS"][-1]))

        summary.append((
            name,
            results["train_Cindex"][-1], results["test_Cindex"][-1],
            results["train_IBS"][-1], results["test_IBS"][-1],
        ))

    # 3. Summary comparison table
    print("=" * 60)
    print("Summary comparison:")
    print("  {:<14s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Partitioning", "TrainC-idx", "TestC-idx", "TrainIBS", "TestIBS"))
    for name, tr_c, te_c, tr_b, te_b in summary:
        print("  {:<14s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            name, tr_c, te_c, tr_b, te_b))

    print("\nMetric interpretation:")
    print("  C-index: higher is better (0.5=random, 1.0=perfect); measures the "
          "ranking ability for survival times")
    print("  IBS:     lower is better (0=perfect); measures the accuracy of "
          "survival probability predictions")
    print("  Partitioning interpretation:")
    print("    IID          - Consistent client distributions; most stable "
          "federated training")
    print("    Non-IID      - Random partitioning, client distributions may shift")
    print("    Time-Non-IID - Partitioning by survival time, simulates time "
          "distribution drift")
    print("    Dirichlet    - alpha controls heterogeneity; can construct complex "
          "non-iid scenarios")

    # 4. Visualization: test C-index over rounds for the 4 partitioning schemes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, results in all_results.items():
        rounds = range(1, len(results["test_Cindex"]) + 1)
        axes[0].plot(rounds, results["test_Cindex"], marker="o", label=name)
        axes[1].plot(rounds, results["test_IBS"], marker="s", label=name)

    axes[0].set_title("Test C-index vs Round (higher is better)")
    axes[0].set_xlabel("Global Round")
    axes[0].set_ylabel("C-index")
    axes[0].set_xticks(range(1, 6))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test IBS vs Round (lower is better)")
    axes[1].set_xlabel("Global Round")
    axes[1].set_ylabel("IBS")
    axes[1].set_xticks(range(1, 6))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "04_partition_comparison.png")
    plt.savefig(out_path, dpi=120)
    print("\nComparison curves saved: {}".format(out_path))

    print("\n=== Example 04 done ===")


if __name__ == "__main__":
    main()
