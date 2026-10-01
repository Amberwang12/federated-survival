# -*- coding: UTF-8 -*-
"""
Example 08: Exponential Mechanism

The exponential mechanism of differential privacy is not for adding noise to
gradients, but for discrete selection problems (e.g. model selection,
hyperparameter picking, client selection). It samples candidates with
exponential probability by quality score: higher-scored items are more
likely to be selected while privacy is preserved.

This example demonstrates:
  1) Private selection among candidate model configurations via the exponential mechanism
  2) Comparing the noise injected by the Gaussian / Laplace mechanisms on a tensor

The closing notes also explain how DP affects federated learning: the
exponential mechanism protects discrete selections without touching gradients
(no training utility cost), unlike gradient-noise mechanisms (see example 07).

API path: federated_survival.core.differential_privacy.DifferentialPrivacy
          .exponential_mechanism / add_gaussian_noise / add_laplace_noise
Run: python examples/08_dp_exponential_mechanism.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import DifferentialPrivacy


def main():
    print("=== Example 08: Exponential Mechanism ===\n")

    np.random.seed(42)

    # Exponential mechanism configuration
    # Note: the larger epsilon is, the sharper the distribution (higher-scored
    # items dominate); here we use 10 so the "high score, high probability"
    # trend is visible within a limited number of samples.
    config = FSAConfig(
        use_differential_privacy=True,
        dp_mechanism="exponential",
        dp_epsilon=10.0,
        dp_sensitivity=1.0,
    )
    dp = DifferentialPrivacy(config)

    # Scenario: 5 candidate model configurations, each with a validation quality score
    # Scores are spread out to make the probability tilt of the exponential mechanism easier to observe
    candidates = torch.randn(5, 20)  # 5 candidate configs, each 20-dimensional
    quality_scores = torch.tensor([0.50, 0.65, 0.90, 0.75, 0.80])
    print("Number of candidate configs: {}".format(len(candidates)))
    print("Quality scores: {}\n".format(quality_scores.tolist()))

    # 1) Sample repeatedly to observe the selection distribution
    #    (higher scores are selected with higher probability, but with randomness)
    print("1) Selection distribution over 3000 samples:")
    counts = np.zeros(5, dtype=int)
    n_trials = 3000
    for _ in range(n_trials):
        idx = dp.exponential_mechanism(candidates, quality_scores)
        counts[idx] += 1
    # Theoretical probability P(i) ∝ exp(eps * q_i / (2 * sensitivity)), for reference
    scores_np = quality_scores.cpu().numpy()
    theory = np.exp(config.dp_epsilon * scores_np / (2 * config.dp_sensitivity))
    theory = theory / theory.sum()
    print("   Candidate  Score   Selected   Theory")
    for i, (s, c) in enumerate(zip(quality_scores.tolist(), counts)):
        print("     {}  {:.2f}   {:6.1%}    {:6.1%}".format(
            i, s, c / n_trials, theory[i]))

    # 2) Single selection, directly returning the selected config tensor
    selected = dp.exponential_mechanism_tensor(candidates, quality_scores)
    print("\n2) Shape of the config tensor returned by a single selection: {}".format(selected.shape))

    # 3) Noise mechanism comparison: add Gaussian / Laplace noise to the same tensor
    print("\n3) Noise mechanism comparison (noise added to a zero tensor):")
    t = torch.zeros(5)

    config_g = FSAConfig(
        use_differential_privacy=True, dp_mechanism="gaussian",
        dp_epsilon=1.0, dp_delta=1e-5, dp_sensitivity=1.0,
        dp_noise_multiplier=1.0, dp_clip_norm=1.0)
    dp_g = DifferentialPrivacy(config_g)
    g = dp_g.add_gaussian_noise(t)

    config_l = FSAConfig(
        use_differential_privacy=True, dp_mechanism="laplace",
        dp_epsilon=1.0, dp_sensitivity=1.0, dp_clip_norm=1.0)
    dp_l = DifferentialPrivacy(config_l)
    l = dp_l.add_laplace_noise(t)

    print("   Original:   {}".format([round(x, 4) for x in t.tolist()]))
    print("   Gaussian:   {}".format([round(x, 4) for x in g.tolist()]))
    print("   Laplace:    {}".format([round(x, 4) for x in l.tolist()]))

    print("\nNotes:")
    print("  Exponential mechanism: discrete selection, probability sampling, no noise added, preserves output semantics")
    print("  Gaussian:   (eps,delta)-DP, normal noise, symmetric, suited to deep learning")
    print("  Laplace:    eps-DP, Laplace noise, heavier tails, suited to numerical queries")
    print("\n  Role of epsilon:")
    print("    Larger eps -> sharper distribution, higher-scored items dominate (weaker privacy)")
    print("    Smaller eps -> flatter distribution, close to uniform sampling (stronger privacy)")
    print("    In practice, trade off between 'privacy strength' and 'selection quality'")
    print("\n  How DP affects federated learning (see example 07 for a measured demo):")
    print("    - The exponential mechanism protects DISCRETE choices (model selection,")
    print("      client selection, hyperparameter picking). It never touches gradients,")
    print("      so it does NOT slow down or degrade federated training itself.")
    print("    - In contrast, applying Gaussian/Laplace noise to client updates (the")
    print("      use_differential_privacy=True path) perturbs every aggregate of every")
    print("      round: T x E noisy updates in total. When the noise scale sigma is")
    print("      comparable to the clipped update norm, the aggregate carries almost no")
    print("      usable gradient signal and the model barely improves (C-index can be")
    print("      frozen at chance level); with small sigma utility is preserved but the")
    print("      true privacy loss can be orders of magnitude above the nominal epsilon.")
    print("    - Rule of thumb: use the exponential mechanism wherever the output is a")
    print("      choice, and reserve gradient noise for when record-level DP of the")
    print("      training data itself is required - and expect a utility price there.")
    print("\n=== Example 08 Done ===")


if __name__ == "__main__":
    main()
