# -*- coding: UTF-8 -*-
"""
Example 10: Reproduce the paper's proof configuration

Paper configuration:
  Simulation: SDGM1, n=100
  Federated: 3 clients, Dirichlet alpha=0.8
  Model: DeepSurv (num_nodes=(32,32))
  Training: 5 global rounds, 20 local epochs, batch_size=32
  Privacy: Gaussian DP, epsilon=1

To show the impact of differential privacy, a "no DP" baseline is also run.

Method path: federated_survival.core.runner.FSARunner
          federated_survival.core.differential_privacy.DifferentialPrivacy
Usage: python examples/10_reproduce_paper.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
# pycox raises RuntimeWarning on model output overflow; known to be caused by small samples + strong DP here, suppressed to keep output clean
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; comment out this line to show windows
import matplotlib.pyplot as plt

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner
from federated_survival.core.differential_privacy import DifferentialPrivacy


# ===== Paper proof configuration =====
SIM = "SDGM1"
N_SAMPLES = 100
N_FEATURES = 10
N_CLIENTS = 3
SPLIT = "Dirichlet"
ALPHA = 0.8
MODEL = "DeepSurv"
NUM_NODES = (32, 32)
GLOBAL_EPOCHS = 5
LOCAL_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DP_MECHANISM = "gaussian"
DP_EPSILON = 1.0
RANDOM_STATE = 42


def make_config(use_dp):
    """Build FSAConfig from the paper configuration"""
    cfg = dict(
        num_clients=N_CLIENTS,
        n_features=N_FEATURES,
        n_samples=N_SAMPLES,
        model_type=MODEL,
        num_nodes=NUM_NODES,
        local_epochs=LOCAL_EPOCHS,
        global_epochs=GLOBAL_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_STATE,
        client_sample_ratio=1.0,
        early_stopping=False,
        verbose=False,
    )
    if use_dp:
        cfg.update(
            use_differential_privacy=True,
            dp_mechanism=DP_MECHANISM,
            dp_epsilon=DP_EPSILON,
            dp_delta=1e-5,            # required by the Gaussian mechanism
            dp_sensitivity=1.0,
            dp_noise_multiplier=1.0,  # required by the Gaussian mechanism
            dp_clip_norm=1.0,
        )
    return FSAConfig(**cfg)


def describe_clients(dataset):
    """Print sample counts and censoring rates per client"""
    for cid, (X, y) in dataset.clients_set.items():
        n = len(y)
        censor_rate = 1.0 - float(np.mean(y[:, 1]))
        print("    {}: n_samples={:<4d} censoring_rate={:.1%}".format(cid, n, censor_rate))


def main():
    print("=" * 64)
    print("Example 10: Reproduce the paper's proof configuration")
    print("=" * 64)

    # 1. Print configuration
    print("\n[Configuration]")
    print("  Simulation:  {} (n={}, d={})".format(SIM, N_SAMPLES, N_FEATURES))
    print("  Split:       {} alpha={}".format(SPLIT, ALPHA))
    print("  Clients:     {}".format(N_CLIENTS))
    print("  Model:       {} num_nodes={}".format(MODEL, NUM_NODES))
    print("  Training:    global={} local={} batch={}".format(
        GLOBAL_EPOCHS, LOCAL_EPOCHS, BATCH_SIZE))
    print("  Privacy:     {} DP, epsilon={}".format(DP_MECHANISM, DP_EPSILON))
    print("  Baseline:    a no-DP run is also performed")

    # 2. Generate data (SDGM1)
    print("\n[1] Generating SDGM1 simulation data...")
    gen = DataGenerator(SimulationConfig(
        n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=RANDOM_STATE))
    data = gen.generate(SIM)
    print("  Data shape: {}, overall censoring rate: {:.1%}, events: {}".format(
        data.shape, 1.0 - float(data["status"].mean()), int(data["status"].sum())))

    # 3. Dirichlet split (alpha=0.8)
    print("\n[2] Dirichlet split (alpha={})...".format(ALPHA))
    splitter = DataSplitter(
        n_clients=N_CLIENTS, split_type=SPLIT, alpha=ALPHA,
        test_size=0.2, random_state=RANDOM_STATE)
    dataset = splitter.split(data)
    print("  Client distribution:")
    describe_clients(dataset)
    print("  Test set shape: {}".format(dataset.test_data.shape))

    # 4. Print DP noise scale (Gaussian mechanism)
    print("\n[3] Gaussian DP noise parameters:")
    dp_cfg = make_config(use_dp=True)
    dp = DifferentialPrivacy(dp_cfg)
    total_eps, per_round_eps = dp.compute_privacy_budget(
        num_rounds=GLOBAL_EPOCHS, num_clients=N_CLIENTS)
    noise_scale = dp.get_noise_scale(num_clients=N_CLIENTS)
    print("    mechanism     = {}".format(dp_cfg.dp_mechanism))
    print("    epsilon       = {}".format(dp_cfg.dp_epsilon))
    print("    delta         = {}".format(dp_cfg.dp_delta))
    print("    sensitivity   = {}".format(dp_cfg.dp_sensitivity))
    print("    clip_norm     = {}".format(dp_cfg.dp_clip_norm))
    print("    total eps     = {:.4f}".format(total_eps))
    print("    per-round eps = {:.4f}".format(per_round_eps))
    print("    noise scale sigma = {:.4f} (with {} clients)".format(
        noise_scale, N_CLIENTS))

    # 5. Training: with DP vs without DP
    results_all = {}
    for label, use_dp in [("with DP (eps=1)", True), ("without DP", False)]:
        print("\n[4] FedAvg training: {} ...".format(label))
        config = make_config(use_dp=use_dp)
        runner = FSARunner(config)
        results = runner.run(dataset, type="raw")
        results_all[label] = results
        print("  train C-index curve: {}".format(
            ["{:.4f}".format(x) for x in results["train_Cindex"]]))
        print("  test C-index curve:  {}".format(
            ["{:.4f}".format(x) for x in results["test_Cindex"]]))
        print("  test IBS curve:      {}".format(
            ["{:.4f}".format(x) for x in results["test_IBS"]]))
        print("  => final test C-index={:.4f}, IBS={:.4f}".format(
            results["test_Cindex"][-1], results["test_IBS"][-1]))

    # 6. Summary comparison
    print("\n" + "=" * 64)
    print("Summary comparison:")
    print("  {:<20s} {:>12s} {:>12s} {:>12s}".format(
        "Experiment", "Train C-idx", "Test C-idx", "Test IBS"))
    for label, res in results_all.items():
        print("  {:<20s} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            label,
            res["train_Cindex"][-1],
            res["test_Cindex"][-1],
            res["test_IBS"][-1]))

    print("\nInterpretation:")
    print("  DP protects privacy via gradient clipping + noise injection, usually at the cost of a slight performance drop")
    print("  With n=100 small samples + 20 local epochs, overfitting is likely; DP noise may instead act as regularization")
    print("  If the two runs differ only slightly, it may be because the model's fitting capacity is limited on small samples")
    # Specific diagnosis for this run's results
    dp_cindex = results_all["with DP (eps=1)"]["test_Cindex"][-1]
    if np.isnan(dp_cindex) or dp_cindex < 0.3:
        print("\n  [Diagnosis] With this configuration, Gaussian DP (eps=1) causes model collapse:")
        print("    - n=100 is an extremely small sample, only ~26 training samples per client")
        print("    - eps=1 gives a large noise scale (sigma~2.8), so the gradient SNR is extremely low")
        print("    - 20 local epochs let the noise accumulate, and model outputs overflow (exp overflow)")
        print("    - Mitigation: increase eps / increase n / reduce local epochs / lower lr")

    # 7. Visualize comparison curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rounds = range(1, GLOBAL_EPOCHS + 1)
    for label, res in results_all.items():
        cindex = np.array(res["test_Cindex"], dtype=float)
        ibs = np.array(res["test_IBS"], dtype=float)
        # C-index: plot directly (zero values are also informative)
        axes[0].plot(rounds, cindex, marker="o", label=label)
        # IBS: break the line at nan
        valid = ~np.isnan(ibs)
        if valid.any():
            axes[1].plot(np.array(list(rounds))[valid], ibs[valid],
                         marker="s", label=label)
        else:
            axes[1].plot([], [], marker="s", label="{} (IBS=nan, model collapsed)".format(label))

    axes[0].set_title("Test C-index vs Round (SDGM1, DeepSurv, Gaussian DP eps=1)")
    axes[0].set_xlabel("Global Round")
    axes[0].set_ylabel("C-index")
    axes[0].set_xticks(list(rounds))
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="random (0.5)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test IBS vs Round")
    axes[1].set_xlabel("Global Round")
    axes[1].set_ylabel("IBS")
    axes[1].set_xticks(list(rounds))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "10_reproduce_paper.png")
    plt.savefig(out_path, dpi=120)
    print("\nComparison curves saved: {}".format(out_path))

    print("\n=== Example 10 Done ===")


if __name__ == "__main__":
    main()
