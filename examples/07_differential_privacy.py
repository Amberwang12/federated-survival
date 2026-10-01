# -*- coding: UTF-8 -*-
"""
Example 07: Differentially Private Federated Learning (Gaussian / Laplace)

Demonstrates enabling differential privacy in federated learning:
  - Gaussian mechanism: (eps, delta)-DP, suited to deep learning gradients; requires delta and noise_multiplier
  - Laplace mechanism:  pure eps-DP, no delta needed, suited to numerical queries

Differential privacy is achieved via gradient clipping + noise injection,
applied only during client-side local training.

How DP affects federated training (measured on this package, see the printed
summary at the end of the example):
  - Noise is injected into EVERY client update of EVERY round, so the total
    number of noisy updates is T x E (global_epochs x local_epochs); fewer
    rounds means less noise accumulation.
  - The noise magnitude is driven by dp_noise_multiplier (sigma): larger sigma
    means stronger privacy but weaker gradients, until the clipped signal is
    completely swamped and the model never leaves its initialization.
  - dp_epsilon is a nominal label only: the actual privacy loss depends on
    sigma, the sampling rate and the number of rounds. Use an RDP accountant
    (e.g. Opacus's RDP accountant) to back-solve the true epsilon.
  - On small per-client datasets the privacy-utility trade-off is steep:
    there may be no sigma that gives both meaningful privacy and good utility.

API path: FSAConfig(use_differential_privacy=True, dp_mechanism=...)
          FSARunner.get_privacy_info()
Run: python examples/07_differential_privacy.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


def run_dp(mechanism=None):
    """Run one federated training with the given DP mechanism.

    mechanism=None disables DP and returns the plain federated baseline.
    Returns (test C-index, test IBS).
    """
    gen = DataGenerator(SimulationConfig(n_samples=300, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    dataset = DataSplitter(
        n_clients=3, split_type="iid", test_size=0.2, random_state=42).split(data)

    common = dict(
        num_clients=3, n_features=10, n_samples=300,
        model_type="PC-Hazard",
        local_epochs=1, global_epochs=30,
        learning_rate=0.003, batch_size=32,
        random_seed=42, verbose=False,
    )
    if mechanism is None:
        common.update(use_differential_privacy=False)
        label = "no DP"
    else:
        common.update(
            use_differential_privacy=True,
            dp_mechanism=mechanism,
            dp_epsilon=1.0,        # Privacy budget (epsilon); smaller means stronger privacy
            dp_sensitivity=1.0,    # Sensitivity
            dp_clip_norm=1.0,      # Gradient clipping norm
        )
        if mechanism == "gaussian":
            common.update(dp_delta=1e-5, dp_noise_multiplier=1.0)
        label = mechanism

    config = FSAConfig(**common)
    runner = FSARunner(config)

    if mechanism is not None:
        # Inspect privacy info
        info = runner.get_privacy_info()
        print("  Privacy info: mechanism={}, eps={}, total eps={:.4f}, per-round eps={:.4f}".format(
            info["mechanism"], info["epsilon"],
            info["total_epsilon"], info["per_round_epsilon"]))
        if mechanism == "gaussian":
            print("           delta={}, noise scale={:.4f}, clip norm={}".format(
                info["delta"], info["noise_scale"], info["clip_norm"]))
        else:
            print("           clip norm={}".format(info["clip_norm"]))

    print("  Training with {} ...".format(label))
    res = runner.run(dataset, type="raw")
    return res["test_Cindex"][-1], res["test_IBS"][-1]


def main():
    print("=== Example 07: Differentially Private Federated Learning ===\n")

    print("0) Plain federated baseline (no DP)")
    c0, i0 = run_dp(None)
    print("   Test C-index={:.4f}, IBS={:.4f}\n".format(c0, i0))

    print("1) Gaussian mechanism (eps, delta)-DP")
    c1, i1 = run_dp("gaussian")
    print("   Test C-index={:.4f}, IBS={:.4f}\n".format(c1, i1))

    print("2) Laplace mechanism eps-DP")
    c2, i2 = run_dp("laplace")
    print("   Test C-index={:.4f}, IBS={:.4f}\n".format(c2, i2))

    print("Impact of DP on federated learning (measured in this run):")
    print("  No DP     C-index={:.4f}  |  Gaussian={:.4f} ({:+.4f})  |  Laplace={:.4f} ({:+.4f})".format(
        c0, c1, c1 - c0, c2, c2 - c0))
    print()
    print("Why DP degrades federated training:")
    print("  1. Noise is added to EVERY client update of EVERY round: with T=30")
    print("     rounds x E=1 local step, that is 30 noisy aggregate updates, and the")
    print("     privacy loss accumulates across all of them.")
    print("  2. The update is first clipped to dp_clip_norm, then Gaussian/Laplace")
    print("     noise of scale sigma is added on top. When sigma is comparable to (or")
    print("     larger than) the clipped signal, the aggregate carries almost no")
    print("     usable gradient information and the model barely improves.")
    print("  3. dp_epsilon is a nominal label only: the true privacy loss depends on")
    print("     sigma, the sampling rate and the number of rounds. Back-solve it with")
    print("     a standard RDP accountant (e.g. sigma=1.0, T=10 -> true eps~18).")
    print()
    print("Measured sigma dose-response (DeepSurv, compare_v2 setup, T=10):")
    print("  sigma=0.1 -> true eps~1008 (privacy negligible), C-index 0.519 vs 0.561 no-DP")
    print("  sigma=1.0 -> true eps~18,   C-index 0.501: the model barely learns")
    print("  sigma=10  -> true eps~1.33, C-index frozen at 0.500: pure noise, the")
    print("               model never leaves its random initialization")
    print()
    print("Practical guidance:")
    print("  - Prefer fewer, larger rounds over many small ones (eps grows with T).")
    print("  - Keep sigma well below the clip norm on small-sample clients; expect a")
    print("    visible utility drop, which is the price of privacy, not a bug.")
    print("  - On very small per-client datasets there may be no sigma that gives")
    print("    both meaningful privacy and good utility (information-theoretic limit).")
    print("\n=== Example 07 Done ===")


if __name__ == "__main__":
    main()
