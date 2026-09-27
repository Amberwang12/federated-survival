"""Focused audit: do dp_clip_norm / dp_noise_multiplier / batch_size change fit?

Uses prediction_validation="warn" so degenerate runs still return metrics
instead of aborting, and reports the diagnostics for each setting.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Ensure the repository source wins over the installed release in site-packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import federated_survival as fs  # noqa: E402

print(f"[pkg] {fs.__file__} v{getattr(fs, '__version__', '?')}")


def run(seed=0, use_dp=True, **overrides):
    data = fs.simulate_data("SDGM1", n_samples=200, n_features=5, censoring=0.4, seed=seed)
    split = fs.partition_data(
        data, n_clients=3, method="iid", test_size=0.2, seed=seed, standardize=True
    )
    kwargs = dict(
        model="DeepSurv",
        protocol="FedAvg",
        global_rounds=5,
        local_steps=3,
        batch_size=32,
        random_state=seed,
        prediction_validation="warn",
        use_differential_privacy=use_dp,
    )
    if use_dp:
        kwargs.update(dp_epsilon=1.0, dp_delta=1e-5)
    kwargs.update(overrides)          # overrides win -> no duplicate kwargs
    est = fs.FederatedSurvival(**kwargs)
    est.fit(split)
    labels = np.asarray(split.test_label, dtype=float)
    m = est.evaluate(split.test_data, labels[:, 0], labels[:, 1])
    d = m["prediction_diagnostics"]
    surv = est.predict_survival(split.test_data).to_numpy(dtype=float)[-1, :]
    return {
        "c_index": round(float(m["c_index"]), 4),
        "finite": bool(d["finite"]),
        "identical_across_samples": bool(d["identical_across_samples"]),
        "pred_head": [round(float(x), 5) for x in surv[:5]],
    }


def main():
    out = {}
    settings = {
        # DP on: clip-norm / noise-multiplier must move the fitted model.
        "dp clip_norm=0.1": dict(use_dp=True, dp_clip_norm=0.1, dp_noise_multiplier=1.0),
        "dp clip_norm=1.0": dict(use_dp=True, dp_clip_norm=1.0, dp_noise_multiplier=1.0),
        "dp clip_norm=10": dict(use_dp=True, dp_clip_norm=10.0, dp_noise_multiplier=1.0),
        "dp noise_mult=0.1": dict(use_dp=True, dp_clip_norm=1.0, dp_noise_multiplier=0.1),
        "dp noise_mult=5": dict(use_dp=True, dp_clip_norm=1.0, dp_noise_multiplier=5.0),
        # DP off: only batch_size differs.
        "batch_size=8": dict(use_dp=False, batch_size=8),
        "batch_size=64": dict(use_dp=False, batch_size=64),
    }
    preds = {}
    for tag, kw in settings.items():
        try:
            res = run(0, **kw)
            out[tag] = res
            preds[tag] = np.asarray(res.pop("pred_head"))
        except Exception as exc:  # noqa: BLE001
            out[tag] = {"ERROR": f"{type(exc).__name__}: {exc}"}
    for a, b, label in (
        ("dp clip_norm=0.1", "dp clip_norm=10", "|clip0.1 - clip10|"),
        ("dp noise_mult=0.1", "dp noise_mult=5", "|noise0.1 - noise5|"),
        ("batch_size=8", "batch_size=64", "|bs8 - bs64|"),
    ):
        if a in preds and b in preds:
            out[f"{label} head"] = list(np.round(np.abs(preds[a] - preds[b]), 6))
    print(json.dumps(out, indent=2))


main()
