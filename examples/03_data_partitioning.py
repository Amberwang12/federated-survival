# -*- coding: UTF-8 -*-
"""
Example 03: Data Partitioning (DataSplitter)

Demonstrates 4 federated data partitioning schemes, with per-client statistics
on sample size, censoring rate, and survival time distribution:
  - iid:          clients have consistent censoring rates (stratified) and
                  similar time distributions
  - non-iid:      equal split after random shuffling; slight shifts in all
                  dimensions are possible
  - time-non-iid: hard split after sorting by survival time; each client
                  covers only one time interval
  - Dirichlet:    time binning + event status form pseudo-classes, allocated
                  by Dirichlet(alpha) sampling; each client covers the full
                  time range, but pseudo-class proportions are heterogeneous
                  (smaller alpha = more non-iid)

Key differences (time-non-iid vs Dirichlet):
  - time-non-iid: hard split on the time dimension; per-client time means
                  increase monotonically and do not overlap
  - Dirichlet:    joint heterogeneity in time + events; each client covers a
                  wide time range, but censoring rates / class proportions
                  differ substantially
  - The difference lies in the time distribution and the degree of censoring
    rate heterogeneity, not just the mean censoring rate

Method path: federated_survival.data.splitter.DataSplitter.split
Run: python examples/03_data_partitioning.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; comment out to show windows
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter


def stat_clients(dataset, title):
    """Print per-client sample size, censoring rate, and time stats (mean/median/range)"""
    print("\n[{}]".format(title))
    print("  {:<10s} {:>5s} {:>8s} {:>10s} {:>10s} {:>16s}".format(
        "client", "n", "censor", "t_mean", "t_median", "t_range"))
    times_list = []
    censors = []
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor = 1.0 - y[:, 1].mean()
        t = y[:, 0]
        times_list.append(t)
        censors.append(censor)
        print("  {:<10s} {:>5d} {:>7.1%} {:>10.2f} {:>10.2f}  [{:.1f}, {:.1f}]".format(
            cid, n, censor, t.mean(), np.median(t), t.min(), t.max()))
    # Heterogeneity metric: std of censoring rates (larger = more non-iid)
    print("  -> Censoring rate std: {:.3f} (larger = more non-iid)".format(np.std(censors)))
    return times_list


def main():
    print("=== Example 03: Data Partitioning ===\n")

    gen = DataGenerator(SimulationConfig(n_samples=600, n_features=10, random_state=42))
    data = gen.generate("SDGM1", c_mean=0.4)
    print("Raw data: {}, overall censoring rate={:.1%}, time range=[{:.1f}, {:.1f}]\n".format(
        data.shape, 1 - data["status"].mean(), data["time"].min(), data["time"].max()))

    all_times = {}
    for split_type in ["iid", "non-iid", "time-non-iid", "Dirichlet"]:
        splitter = DataSplitter(
            n_clients=4,
            split_type=split_type,
            alpha=0.3,        # Dirichlet parameter; smaller = more non-iid
            test_size=0.2,
            random_state=42,
        )
        dataset = splitter.split(data)
        all_times[split_type] = stat_clients(dataset, split_type)

    # Visualization: box plots of per-client time distributions under 4 split types
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (split_type, times_list) in zip(axes, all_times.items()):
        bp = ax.boxplot(times_list, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(split_type)
        ax.set_xlabel("client id")
        ax.set_xticklabels(["c0", "c1", "c2", "c3"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("survival time")
    fig.suptitle("Survival time distribution per client (4 split types)", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "03_partition_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print("\nTime distribution comparison plot saved: {}".format(out_path))

    print("\nNotes:")
    print("  IID          - Consistent censoring rates and similar time "
          "distributions across clients (ideal federated setting)")
    print("  Non-IID      - Random partitioning, slight shifts in all dimensions")
    print("  Time-Non-IID - Hard split after sorting by time; per-client time "
          "means increase monotonically (c0=short, c3=long)")
    print("                 Suitable for simulating time distribution drift "
          "(e.g. hospitals admitting different disease stages)")
    print("  Dirichlet    - Time binning + event status form pseudo-classes, "
          "sampled with Dirichlet(alpha)")
    print("                 Each client covers the full time range, but "
          "censoring rates / class proportions are heterogeneous")
    print("                 Smaller alpha = more non-iid (censoring rate std is "
          "largest at alpha=0.3)")
    print("\n  Key difference: time-non-iid is a hard time split; Dirichlet is "
          "joint soft heterogeneity in time + events.")

    print("\n=== Example 03 done ===")


if __name__ == "__main__":
    main()
