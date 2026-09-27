"""End-to-end real/simulated experiment execution from a normalized config."""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
import yaml

from ..core.config import FSAConfig
from ..data.generator import DataGenerator, SimulationConfig
from ..data.preprocessing import preprocess_partition
from ..data.splitter import DataSet, DataSplitter
from .baselines import run_paired_baselines
from .config_schema import normalize_workflow_config
from .infrastructure import ExperimentRecorder, environment_report, write_json

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
}
PROTOCOL_PARAMETER_ALIASES = {
    "mu": "proximal_mu",
    "optimizer": "server_optimizer",
    "server_lr": "server_learning_rate",
    "beta1": "server_beta1",
    "beta2": "server_beta2",
    "tau": "server_tau",
    "max_iter": "webdisco_max_iter",
    "tolerance": "webdisco_tolerance",
    "l2": "webdisco_l2",
    "ridge": "webdisco_ridge",
    "max_step_norm": "webdisco_max_step_norm",
}


def _read_input(data: Dict[str, Any], seed: int) -> pd.DataFrame:
    source = data["source"]
    if source == "simulation":
        return DataGenerator(
            SimulationConfig(
                n_samples=data["n_samples"],
                n_features=data["n_features"],
                random_state=seed,
            )
        ).generate(data["mechanism"], c_mean=data["censoring_rate"])

    path = Path(data["resolved_path"])
    frame = pd.read_csv(path) if source == "csv" else pd.read_excel(path)
    duration = data["duration_column"]
    event = data["event_column"]
    if duration == event:
        raise ValueError("duration and event columns must be different")
    missing_columns = {duration, event} - set(frame.columns)
    if missing_columns:
        raise ValueError("data is missing required columns: %s" % sorted(missing_columns))
    if frame.columns.duplicated().any():
        raise ValueError("data contains duplicate column names")

    feature_columns = [column for column in frame.columns if column not in (duration, event)]
    if not feature_columns:
        raise ValueError("data must contain at least one feature column")
    canonical = frame[feature_columns + [duration, event]].copy()
    canonical = canonical.rename(columns={duration: "time", event: "status"})
    for column in feature_columns + ["time", "status"]:
        canonical[column] = pd.to_numeric(canonical[column], errors="raise")
    if canonical[["time", "status"]].isna().any().any():
        raise ValueError("duration and event columns cannot contain missing values")

    missing_policy = data["missing_values"]
    if missing_policy == "error" and canonical[feature_columns].isna().any().any():
        raise ValueError("feature columns contain missing values; choose drop or median")
    if missing_policy == "drop":
        canonical = canonical.dropna(axis=0).reset_index(drop=True)
    if not np.isfinite(canonical[["time", "status"]].to_numpy(dtype=float)).all():
        raise ValueError("duration and event columns must be finite")
    if (canonical["time"] <= 0).any():
        raise ValueError("all observed times must be positive")
    observed_statuses = set(canonical["status"].astype(float).unique())
    if not observed_statuses.issubset({0.0, 1.0}):
        raise ValueError("event column must contain only zero and one")
    canonical["status"] = canonical["status"].astype(float)
    return canonical


def _partition_rows(dataset: DataSet, seed: int):
    summaries, observations = [], []
    for client, (features, labels) in dataset.clients_set.items():
        features = np.asarray(features)
        labels = np.asarray(labels)
        status = labels[:, 1]
        summaries.append(
            {
                "seed": seed,
                "client": client,
                "n": len(labels),
                "events": int(status.sum()),
                "censored": int(len(labels) - status.sum()),
                "censoring_rate": float(1.0 - status.mean()),
                "time_min": float(labels[:, 0].min()),
                "time_median": float(np.median(labels[:, 0])),
                "time_max": float(labels[:, 0].max()),
            }
        )
        for index, (feature_row, label_row) in enumerate(zip(features, labels)):
            observation = {
                "seed": seed,
                "client": client,
                "sample_index": index,
                "time": float(label_row[0]),
                "status": int(label_row[1]),
            }
            observation.update(
                {
                    "feature_%d" % (feature_index + 1): float(value)
                    for feature_index, value in enumerate(feature_row)
                }
            )
            observations.append(observation)
    return summaries, observations


def _data_summary(frame: pd.DataFrame, seed: int) -> Dict[str, Any]:
    return {
        "seed": seed,
        "n_rows": int(len(frame)),
        "n_features": int(frame.shape[1] - 2),
        "events": int(frame["status"].sum()),
        "censored": int(len(frame) - frame["status"].sum()),
        "censoring_rate": float(1.0 - frame["status"].mean()),
        "time_min": float(frame["time"].min()),
        "time_median": float(frame["time"].median()),
        "time_max": float(frame["time"].max()),
    }


def _fsa_config(configuration: Dict[str, Any], frame: pd.DataFrame, seed: int) -> FSAConfig:
    data = configuration["data"]
    partition = configuration["partition"]
    model = configuration["model"]
    federated = configuration["federated"]
    evaluation = configuration["evaluation"]
    values = {
        "mode": "real" if data["source"] != "simulation" else "simulate",
        "dataset_name": configuration["name"] if data["source"] != "simulation" else None,
        "n_samples": len(frame),
        "n_features": frame.shape[1] - 2,
        "model_type": model["name"],
        "num_nodes": tuple(model["hidden_nodes"]),
        "num_durations": model["num_durations"],
        "activation": ACTIVATIONS[model["activation"]],
        "dropout": model["dropout"],
        "weight_decay": model["weight_decay"],
        "num_clients": partition["n_clients"],
        "global_epochs": federated["global_rounds"],
        "local_epochs": federated["local_steps"],
        "batch_size": federated["batch_size"],
        "full_batch": federated["batch_mode"] == "full_batch",
        "learning_rate": federated["learning_rate"],
        "optimizer": federated["optimizer"],
        "client_sample_ratio": federated["client_fraction"],
        "split_method": partition["method"],
        "split_alpha": partition["alpha"],
        "test_size": data["test_size"],
        "random_seed": seed,
        "federated_protocol": federated["protocol"],
        "evaluation_quantiles": tuple(evaluation["time_grid_quantiles"]),
        "cox_loss_normalization": "patient",
        "prediction_validation": "raise",
        "early_stopping": False,
        "show_progress": False,
        "verbose": False,
    }
    for key, value in federated["protocol_params"].items():
        values[PROTOCOL_PARAMETER_ALIASES.get(key, key)] = value
    return FSAConfig(**values)


def _summary_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    selected = raw[raw["method"].isin(["Center", "FSA", "Local"])].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["protocol", "model", "method"], sort=False)
        .agg(
            n=("seed", "nunique"),
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            c_index_median=("c_index", "median"),
            c_index_q1=("c_index", lambda value: value.quantile(0.25)),
            c_index_q3=("c_index", lambda value: value.quantile(0.75)),
            ibs_mean=("ibs", "mean"),
            ibs_sd=("ibs", "std"),
            ibs_median=("ibs", "median"),
            ibs_q1=("ibs", lambda value: value.quantile(0.25)),
            ibs_q3=("ibs", lambda value: value.quantile(0.75)),
        )
        .reset_index()
    )


def _write_yaml(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _record_artifact(recorder: ExperimentRecorder, path: Path, kind: str) -> None:
    if path.exists():
        recorder.add_artifact(path, kind)


def run_workflow_config(
    configuration: Dict[str, Any],
    config_path: Path,
    output_override: Optional[Path] = None,
) -> Path:
    """Execute a schema-v1 experiment and produce data, metrics, audit files, and plots."""
    normalized = normalize_workflow_config(configuration, config_path)
    output_dir = Path(output_override or normalized["output"]["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not normalized["output"]["overwrite"]:
        raise FileExistsError(
            "output directory is not empty; choose another directory or set "
            "output.overwrite: true: %s" % output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = ExperimentRecorder(output_dir, normalized["name"], normalized)

    resolved_path = output_dir / "resolved_config.yaml"
    _write_yaml(resolved_path, normalized)
    environment_path = output_dir / "environment.json"
    write_json(environment_path, environment_report())

    raw_frames: List[pd.DataFrame] = []
    telemetry_rows: List[Dict[str, Any]] = []
    prediction_rows: Optional[List[Dict[str, Any]]] = (
        [] if normalized["evaluation"]["save_predictions"] else None
    )
    partition_summaries: List[Dict[str, Any]] = []
    partition_observations: List[Dict[str, Any]] = []
    data_summaries: List[Dict[str, Any]] = []
    simulated_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, Any]] = []

    for seed in normalized["experiment"]["seeds"]:
        try:
            frame = _read_input(normalized["data"], seed)
            data_summaries.append(_data_summary(frame, seed))
            if normalized["data"]["source"] == "simulation":
                saved_frame = frame.copy()
                saved_frame.insert(0, "seed", seed)
                simulated_frames.append(saved_frame)
            dataset = DataSplitter(
                n_clients=normalized["partition"]["n_clients"],
                split_type=normalized["partition"]["method"],
                alpha=normalized["partition"]["alpha"],
                test_size=normalized["data"]["test_size"],
                random_state=seed,
            ).split(frame)
            dataset = preprocess_partition(
                dataset,
                missing_values=normalized["data"]["missing_values"],
                standardize=normalized["data"]["standardize"],
            )
            summaries, observations = _partition_rows(dataset, seed)
            partition_summaries.extend(summaries)
            partition_observations.extend(observations)
            if normalized["experiment"]["stage"] == "data_only":
                continue
            config = _fsa_config(normalized, frame, seed)
            seed_telemetry: Optional[List[Dict[str, Any]]] = (
                [] if normalized["evaluation"]["save_round_metrics"] else None
            )
            result = run_paired_baselines(
                config,
                dataset,
                seed,
                baseline_epochs=normalized["references"]["optimizer_steps"],
                telemetry_rows=seed_telemetry,
                prediction_rows=prediction_rows,
                include_center=normalized["references"]["centralized"],
                include_local=normalized["references"]["local_models"],
            )
            result["protocol"] = normalized["federated"]["protocol"]
            result["split"] = normalized["partition"]["method"]
            raw_frames.append(result)
            for row in seed_telemetry or []:
                row["protocol"] = normalized["federated"]["protocol"]
                row["split"] = normalized["partition"]["method"]
                telemetry_rows.append(row)
        except Exception as error:
            failures.append(
                {
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            if normalized["experiment"]["fail_fast"]:
                break

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    rounds = pd.DataFrame(telemetry_rows)
    partitions = pd.DataFrame(partition_summaries)
    observations = pd.DataFrame(partition_observations)
    simulated = (
        pd.concat(simulated_frames, ignore_index=True) if simulated_frames else pd.DataFrame()
    )
    predictions = pd.DataFrame(prediction_rows or [])
    failures_frame = pd.DataFrame(failures, columns=["seed", "error_type", "error", "traceback"])

    paths = {
        "data_summary": output_dir / "data_summary.json",
        "partition": output_dir / "client_partition.csv",
        "observations": output_dir / "client_partition_observations.csv",
        "simulated": output_dir / "simulated_data.csv",
        "raw": output_dir / "raw_results.csv",
        "summary": output_dir / "summary_metrics.csv",
        "rounds": output_dir / "round_metrics.csv",
        "predictions": output_dir / "survival_predictions.csv",
        "failures": output_dir / "failures.csv",
        "status": output_dir / "run_status.json",
    }
    write_json(
        paths["data_summary"],
        {
            "source": normalized["data"]["source"],
            "duration_column": normalized["data"]["duration_column"],
            "event_column": normalized["data"]["event_column"],
            "per_seed": data_summaries,
        },
    )
    partitions.to_csv(paths["partition"], index=False)
    observations.to_csv(paths["observations"], index=False)
    if not simulated.empty:
        simulated.to_csv(paths["simulated"], index=False)
    failures_frame.to_csv(paths["failures"], index=False)
    if not raw.empty:
        raw.to_csv(paths["raw"], index=False)
        _summary_metrics(raw).to_csv(paths["summary"], index=False)
    if not rounds.empty:
        rounds.to_csv(paths["rounds"], index=False)
    if not predictions.empty:
        predictions.to_csv(paths["predictions"], index=False)

    figure_paths: List[Path] = []
    if not failures and not observations.empty and normalized["visualization"]["enabled"]:
        # Import plotting only for runs that request/use the completed result;
        # diagnostics and config inspection should not initialize Matplotlib.
        from .visualization import generate_visualizations

        figure_paths = generate_visualizations(raw, rounds, observations, normalized, output_dir)

    expected = len(normalized["experiment"]["seeds"])
    if normalized["experiment"]["stage"] == "data_only":
        completed = int(partitions["seed"].nunique()) if not partitions.empty else 0
    else:
        completed = int(raw["seed"].nunique()) if not raw.empty else 0
    status = {
        "expected_seeds": expected,
        "complete_seeds": completed,
        "failed_seeds": len(failures),
        "metric_nan_rows": (
            int(raw[["c_index", "ibs"]].isna().any(axis=1).sum()) if not raw.empty else 0
        ),
        "complete": completed == expected and not failures,
        "figures": [str(path.resolve()) for path in figure_paths],
    }
    write_json(paths["status"], status)

    for path, kind in (
        (resolved_path, "configuration"),
        (environment_path, "environment"),
        (paths["data_summary"], "data_summary"),
        (paths["partition"], "client_partition"),
        (paths["observations"], "client_partition_observations"),
        (paths["simulated"], "simulated_data"),
        (paths["raw"], "metrics"),
        (paths["summary"], "metric_summary"),
        (paths["rounds"], "round_metrics"),
        (paths["predictions"], "predictions"),
        (paths["failures"], "failure_audit"),
        (paths["status"], "run_status"),
    ):
        _record_artifact(recorder, path, kind)
    for path in figure_paths:
        recorder.add_artifact(path, "figure")
    recorder.finish("complete" if status["complete"] else "partial")

    if failures:
        raise RuntimeError(
            "%d experiment seeds failed; inspect %s" % (len(failures), paths["failures"])
        )
    return output_dir
