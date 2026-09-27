"""Paired Center, FSA, and Local baselines.

The functions in this module deliberately exclude augmentation and differential
privacy.  All methods receive the same train/test split and initialization seed,
so their differences can be analysed as paired observations.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pycox.evaluation import EvalSurv

from federated_survival.core.config import FSAConfig
from federated_survival.core._losses import apply_cox_patient_normalization
from federated_survival.core._minibatch import deterministic_stream_seed, fit_in_local_steps
from federated_survival.core.runner import FSARunner
from federated_survival.core.server import Server
from federated_survival.models import get_model_adapter
from federated_survival.utils.metrics import evaluation_time_grid

MODEL_NAMES = (
    "CoxPH",
    "DeepSurv",
    "CoxCC",
    "CoxTime",
    "LogisticHazard",
    "PC-Hazard",
    "DeepHit",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _targets(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return y[:, 0], y[:, 1]


def _prepare(config: FSAConfig, y: np.ndarray):
    """Fit the adapter-owned label transform and report the output width.

    Delegating to :meth:`ModelAdapter.configure_targets` keeps the centralized
    baselines and the federated runner on a single implementation, so an
    adapter registered through the public ``register_model_adapter`` registry
    behaves identically on both paths.
    """
    adapter = get_model_adapter(config.model_type)
    durations, events = _targets(y)
    labtrans = adapter.configure_targets(config, durations, events)
    transformed = adapter.transform_target(y, labtrans)
    return labtrans, transformed


def _wrap_model(config: FSAConfig, net, optimizer, labtrans):
    """Wrap ``net`` with the pycox model owned by the registered adapter."""
    return get_model_adapter(config.model_type).build_model(
        net, config, label_transform=labtrans, optimizer=optimizer
    )


def _evaluate(model, x_test: np.ndarray, y_test: np.ndarray, quantiles) -> Tuple[float, float]:
    durations, events = _targets(y_test)
    survival = model.predict_surv_df(x_test.astype("float32"))
    evaluator = EvalSurv(survival, durations, events, censor_surv="km")
    grid = evaluation_time_grid(durations, survival.index.values, quantiles)
    return (
        float(evaluator.concordance_td()),
        float(evaluator.integrated_brier_score(grid)),
    )


def _fit_one(
    config: FSAConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit one centralized/local model with an explicit or FSA-matched budget."""
    if np.sum(y_train[:, 1]) == 0:
        raise ValueError("Every training partition must contain at least one event")

    _seed_everything(seed)
    conf = deepcopy(config)
    labtrans, transformed = _prepare(conf, y_train)
    net = Server(conf).global_model
    optimizer_class = torch.optim.SGD if conf.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_class(
        net.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
    )
    model = _wrap_model(conf, net, optimizer, labtrans)
    apply_cox_patient_normalization(model, conf, float(np.mean(y_train[:, 1])))
    optimization_steps = epochs if epochs is not None else conf.global_epochs * conf.local_epochs
    fit_in_local_steps(
        model,
        x_train.astype("float32"),
        transformed,
        model_type=conf.model_type,
        batch_size=conf.batch_size,
        steps=optimization_steps,
        full_batch=conf.full_batch,
        seed=deterministic_stream_seed(seed, f"baseline:{conf.model_type}:{len(x_train)}"),
    )
    if get_model_adapter(conf.model_type).requires_baseline_hazards:
        model.compute_baseline_hazards()
    return _evaluate(model, x_test, y_test, conf.evaluation_quantiles)


def run_paired_baselines(
    config: FSAConfig,
    dataset,
    seed: int,
    baseline_epochs: Optional[int] = None,
    telemetry_rows: Optional[List[Dict[str, object]]] = None,
    prediction_rows: Optional[List[Dict[str, object]]] = None,
    include_center: bool = True,
    include_local: bool = True,
) -> pd.DataFrame:
    """Run Center, FSA, and Local models on one fixed data partition.

    ``baseline_epochs=None`` matches the exact Center/Local optimizer-step
    budget to ``T * E``.  The historical argument name is retained for API
    compatibility.  An
    explicit value is useful for reproducing a published protocol whose
    centralized/local budget differs from its federated budget; such runs
    should be accompanied by the matched-budget sensitivity analysis.
    """
    rows: List[Dict[str, object]] = []

    if include_center:
        center_c, center_ibs = _fit_one(
            config,
            dataset.train_data,
            dataset.train_label,
            dataset.test_data,
            dataset.test_label,
            seed,
            baseline_epochs,
        )
        rows.append(
            {
                "seed": seed,
                "model": config.model_type,
                "method": "Center",
                "client": "all",
                "n_train": len(dataset.train_data),
                "c_index": center_c,
                "ibs": center_ibs,
            }
        )

    _seed_everything(seed)
    runner = FSARunner(deepcopy(config))
    result = runner.run(dataset)
    if telemetry_rows is not None:
        telemetry_flags = {
            key: value for key, value in result.items() if key.startswith("telemetry_is_")
        }
        for round_index in range(len(result["test_Cindex"])):
            telemetry_rows.append(
                {
                    "seed": seed,
                    "model": config.model_type,
                    "round": round_index + 1,
                    "train_c_index": float(result["train_Cindex"][round_index]),
                    "train_ibs": float(result["train_IBS"][round_index]),
                    "test_c_index": float(result["test_Cindex"][round_index]),
                    "test_ibs": float(result["test_IBS"][round_index]),
                    "train_loss": float(result["train_loss"][round_index]),
                    "update_direction_norm": float(result["update_direction_norm"][round_index]),
                    "client_drift": float(result["client_drift"][round_index]),
                    "update_direction_dispersion": float(
                        result["update_direction_dispersion"][round_index]
                    ),
                    "communication_bytes": int(result["communication_bytes"][round_index]),
                    "selected_clients": json.dumps(
                        result["selected_clients"][round_index], ensure_ascii=False
                    ),
                    **telemetry_flags,
                }
            )
    rows.append(
        {
            "seed": seed,
            "model": config.model_type,
            "method": "FSA",
            "client": "all",
            "n_train": len(dataset.train_data),
            "c_index": float(result["test_Cindex"][-1]),
            "ibs": float(result["test_IBS"][-1]),
        }
    )

    if prediction_rows is not None:
        survival = runner.predict_survival(dataset.test_data)
        for sample_index, column in enumerate(survival.columns):
            for time, probability in survival[column].items():
                prediction_rows.append(
                    {
                        "seed": seed,
                        "model": config.model_type,
                        "method": "FSA",
                        "sample_id": int(sample_index),
                        "time": float(time),
                        "survival_probability": float(probability),
                    }
                )

    local_metrics = []
    if include_local:
        for client_id, (x_local, y_local) in dataset.clients_set.items():
            c_index, ibs = _fit_one(
                config,
                x_local,
                y_local,
                dataset.test_data,
                dataset.test_label,
                seed,
                baseline_epochs,
            )
            local_metrics.append((len(x_local), c_index, ibs))
            rows.append(
                {
                    "seed": seed,
                    "model": config.model_type,
                    "method": "Local-client",
                    "client": client_id,
                    "n_train": len(x_local),
                    "c_index": c_index,
                    "ibs": ibs,
                }
            )

        weights = np.asarray([item[0] for item in local_metrics], dtype=float)
        weights /= weights.sum()
        rows.append(
            {
                "seed": seed,
                "model": config.model_type,
                "method": "Local",
                "client": "sample-weighted-mean",
                "n_train": len(dataset.train_data),
                "c_index": float(np.sum(weights * [item[1] for item in local_metrics])),
                "ibs": float(np.sum(weights * [item[2] for item in local_metrics])),
            }
        )
    return pd.DataFrame(rows)


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive paired summaries; no inference is claimed for smoke runs.

    Expects the ``raw_results.csv`` layout produced by
    :mod:`federated_survival.experiments.workflow_runner`: one row per
    ``(split, model, method, seed)``, where ``split`` and ``model`` identify the
    comparison cell and ``method`` is one of ``Center``, ``FSA`` or ``Local``.
    Per-client ``Local-client`` rows are ignored.  Note that
    :func:`run_paired_baselines` omits the ``split`` column -- it is added by
    the workflow runner when the results are persisted.
    """
    aggregate = raw[raw["method"].isin(["Center", "FSA", "Local"])]
    summary = aggregate.groupby(["split", "model", "method"], as_index=False).agg(
        n=("seed", "size"),
        c_index_mean=("c_index", "mean"),
        c_index_sd=("c_index", "std"),
        ibs_mean=("ibs", "mean"),
        ibs_sd=("ibs", "std"),
    )
    wide = aggregate.pivot_table(
        index=["split", "model", "seed"],
        columns="method",
        values=["c_index", "ibs"],
        aggfunc="first",
    )
    deltas = []
    for index, row in wide.iterrows():
        deltas.append(
            {
                "split": index[0],
                "model": index[1],
                "seed": index[2],
                "fsa_minus_center_c_index": row[("c_index", "FSA")] - row[("c_index", "Center")],
                "fsa_minus_local_c_index": row[("c_index", "FSA")] - row[("c_index", "Local")],
                "fsa_minus_center_ibs": row[("ibs", "FSA")] - row[("ibs", "Center")],
                "fsa_minus_local_ibs": row[("ibs", "FSA")] - row[("ibs", "Local")],
            }
        )
    delta_frame = pd.DataFrame(deltas)
    delta_summary = delta_frame.groupby(["split", "model"], as_index=False).agg(
        fsa_minus_center_c_index=("fsa_minus_center_c_index", "mean"),
        fsa_minus_local_c_index=("fsa_minus_local_c_index", "mean"),
        fsa_minus_center_ibs=("fsa_minus_center_ibs", "mean"),
        fsa_minus_local_ibs=("fsa_minus_local_ibs", "mean"),
    )
    return summary.merge(delta_summary, on=["split", "model"], how="left")
