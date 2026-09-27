"""Ad-hoc audit script: verify the reviewer's software/empirical complaints.

Runs the manuscript's headline regime (SDGM1, small n, 3 clients) across
several seeds and checks that (a) predictions are usable, (b) metrics are
above chance, (c) exposed DP parameters and batch_size actually change the
fitted model, and (d) the documented calculate_ibs() works.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

# Make sure the *repository* source wins over any installed release
# (the py38 environment ships the published 0.5.0 in site-packages).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import federated_survival as fs  # noqa: E402
from federated_survival.utils.metrics import (  # noqa: E402
    calculate_ibs,
    evaluation_time_grid,
)

print(f"[pkg] {fs.__file__} v{getattr(fs, '__version__', '?')}")

RESULTS = {}


def report(key, value):
    RESULTS[key] = value
    print(f"[{key}] {value}")


def _survival_matrix(predictions, method="FSA"):
    frame = predictions[predictions["method"] == method]
    wide = frame.pivot_table(
        index="time", columns="sample_id", values="survival_probability", aggfunc="first"
    )
    return wide.to_numpy(dtype=float)


# ---------------------------------------------------------------- 1. headline
def headline(seeds=(0, 1, 2, 3, 4)):
    rows = []
    for seed in seeds:
        data = fs.simulate_data(
            mechanism="SDGM1", n_samples=100, n_features=5, censoring=0.4, seed=seed
        )
        split = fs.partition_data(
            data, n_clients=3, method="iid", test_size=0.2, seed=seed, standardize=True
        )
        res = fs.compare_methods(
            split,
            model="DeepSurv",
            protocol="FedAvg",
            global_rounds=10,
            local_steps=5,
            reference_steps=50,
            batch_size=32,
            seed=seed,
            save_predictions=True,
        )
        m = res.metrics
        vals = _survival_matrix(res.predictions, "FSA")
        rows.append(
            {
                "seed": seed,
                "federated_cindex": float(m.loc[m["method"] == "Federated", "c_index"].iloc[0]),
                "center_cindex": float(m.loc[m["method"] == "Center", "c_index"].iloc[0]),
                "local_cindex": float(m.loc[m["method"] == "Weighted Local", "c_index"].iloc[0]),
                "finite": bool(np.isfinite(vals).all()),
                "n_unique_curves": int(np.unique(np.round(vals, 8), axis=1).shape[1]),
                "n_samples": int(vals.shape[1]),
                "monotone": bool(np.all(np.diff(vals, axis=0) <= 1e-6)),
            }
        )
    report("headline_rows", rows)
    c = [r["federated_cindex"] for r in rows]
    report("headline_federated_cindex_mean", float(np.mean(c)))
    report("headline_federated_cindex_sd", float(np.std(c, ddof=1)))
    report("headline_all_finite", all(r["finite"] for r in rows))
    report("headline_all_curves_distinct", all(r["n_unique_curves"] > 1 for r in rows))
    report("headline_above_chance_all_seeds", all(x > 0.5 for x in c))


# ------------------------------------------------- 2. parameter effectiveness
def _dp_run(seed, **overrides):
    data = fs.simulate_data(
        mechanism="SDGM1", n_samples=200, n_features=5, censoring=0.4, seed=seed
    )
    split = fs.partition_data(
        data, n_clients=3, method="iid", test_size=0.2, seed=seed, standardize=True
    )
    est = fs.FederatedSurvival(
        model="DeepSurv",
        protocol="FedAvg",
        global_rounds=5,
        local_steps=3,
        batch_size=32,
        random_state=seed,
        use_differential_privacy=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        **overrides,
    )
    est.fit(split)
    labels = np.asarray(split.test_label, dtype=float)
    metrics = est.evaluate(split.test_data, labels[:, 0], labels[:, 1])
    surv = est.predict_survival(split.test_data).to_numpy(dtype=float)[-1, :]
    return float(metrics["c_index"]), np.asarray(surv, dtype=float)


def dp_effectiveness():
    out = {}
    for tag, kw in {
        "clip_norm=0.1": dict(dp_clip_norm=0.1, dp_noise_multiplier=1.0),
        "clip_norm=10": dict(dp_clip_norm=10.0, dp_noise_multiplier=1.0),
        "noise_mult=0.1": dict(dp_clip_norm=1.0, dp_noise_multiplier=0.1),
        "noise_mult=5": dict(dp_clip_norm=1.0, dp_noise_multiplier=5.0),
    }.items():
        cidx, pred = _dp_run(0, **kw)
        out[tag] = {"c_index": cidx, "pred": pred}
    report(
        "dp_clip_norm_changes_predictor",
        float(np.max(np.abs(out["clip_norm=0.1"]["pred"] - out["clip_norm=10"]["pred"]))),
    )
    report(
        "dp_noise_multiplier_changes_predictor",
        float(np.max(np.abs(out["noise_mult=0.1"]["pred"] - out["noise_mult=5"]["pred"]))),
    )

    # batch_size effect (no DP, so only batch_size differs)
    b = {}
    for bs in (8, 64):
        data = fs.simulate_data(
            mechanism="SDGM1", n_samples=200, n_features=5, censoring=0.4, seed=0
        )
        split = fs.partition_data(
            data, n_clients=3, method="iid", test_size=0.2, seed=0, standardize=True
        )
        est = fs.FederatedSurvival(
            model="DeepSurv",
            protocol="FedAvg",
            global_rounds=5,
            local_steps=3,
            batch_size=bs,
            random_state=0,
        )
        est.fit(split)
        b[bs] = est.predict_survival(split.test_data).to_numpy(dtype=float)[-1, :]
    report("batch_size_changes_predictor", float(np.max(np.abs(b[8] - b[64]))))


# ----------------------------------------------------------- 3. Dirichlet path
def dirichlet_path():
    data = fs.simulate_data(mechanism="SDGM1", n_samples=300, n_features=5, seed=0)
    split = fs.partition_data(
        data, n_clients=3, method="dirichlet", test_size=0.2, seed=0, alpha=0.5
    )
    sizes = [len(features) for features, _ in split.clients_set.values()]
    means = [float(np.mean(labels[:, 0])) for _, labels in split.clients_set.values()]
    report("dirichlet_client_sizes", sizes)
    report("dirichlet_client_mean_times", [round(x, 3) for x in means])
    report("dirichlet_is_non_iid", bool(max(means) - min(means) > 0.05))


# ------------------------------------------------------------- 4. calculate_ibs
def ibs_check():
    rng = np.random.RandomState(0)
    n, t = 200, 50
    time = rng.exponential(2.0, n)
    event = (rng.uniform(size=n) > 0.3).astype(float)
    grid = evaluation_time_grid(time, points=t)
    good = np.tile(np.exp(-0.2 * grid)[:, None], (1, n))
    bad = np.tile(np.exp(-2.0 * grid)[:, None], (1, n))
    report("ibs_good_curve", float(calculate_ibs(grid, good, time, event)))
    report("ibs_bad_curve", float(calculate_ibs(grid, bad, time, event)))


def main():
    for name, fn in (
        ("headline", headline),
        ("dp_effectiveness", dp_effectiveness),
        ("dirichlet_path", dirichlet_path),
        ("ibs_check", ibs_check),
    ):
        try:
            fn()
        except Exception:  # noqa: BLE001
            report(f"{name}_ERROR", traceback.format_exc())
    print("\n=====JSON=====")
    print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
