"""Configuration-driven benchmark runner for the public CLI."""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any, Dict

import pandas as pd

from ..api import FederatedSurvival
from ..data import DataGenerator, SimulationConfig
from .config_schema import is_workflow_configuration
from .infrastructure import ExperimentRecorder, load_experiment_config, write_json
from .workflow_runner import run_workflow_config

DEFAULT_MODELS = [
    "CoxPH",
    "DeepSurv",
    "CoxCC",
    "CoxTime",
    "LogisticHazard",
    "PC-Hazard",
    "DeepHit",
]


def _protocol_specs(configuration, federated):
    """Normalize one or more protocol declarations from JSON/YAML."""
    declared = configuration.get("protocols")
    if declared is None:
        return [
            {
                "name": federated.get("protocol", "FedAvg"),
                "params": federated.get("protocol_params", {}),
                "models": None,
            }
        ]
    if not isinstance(declared, list) or not declared:
        raise ValueError("protocols must be a non-empty list")
    normalized = []
    for item in declared:
        if isinstance(item, str):
            normalized.append({"name": item, "params": {}, "models": None})
            continue
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each protocol must be a name or an object with a name")
        models = item.get("models")
        if models is not None and (not isinstance(models, list) or not models):
            raise ValueError("protocol models must be a non-empty list when provided")
        normalized.append(
            {
                "name": item["name"],
                "params": item.get("params", {}),
                "models": models,
            }
        )
    return normalized


def _run_legacy_config(
    configuration: Dict[str, Any],
    config_path: Path,
    output_override: Path = None,
) -> Path:
    """Run a small, auditable multi-model simulation from JSON or YAML."""
    output_dir = Path(
        output_override or configuration.get("output", "results/configured_benchmark")
    )
    recorder = ExperimentRecorder(output_dir, configuration.get("name", "benchmark"), configuration)
    frozen_config = output_dir / "resolved_config.json"
    write_json(frozen_config, configuration)
    recorder.add_artifact(frozen_config, "configuration")

    data_config = configuration.get("data", {})
    federated = configuration.get("federated", {})
    models = configuration.get("models", DEFAULT_MODELS)
    seeds = configuration.get("seeds", [42])
    protocols = _protocol_specs(configuration, federated)
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")

    rows, failures = [], []
    for protocol_spec in protocols:
        protocol_name = protocol_spec["name"]
        protocol_models = protocol_spec["models"] or models
        unknown = sorted(set(protocol_models) - set(models))
        if unknown:
            raise ValueError(
                f"protocol {protocol_name!r} names models not present in models: {unknown}"
            )
        for model in protocol_models:
            for seed in seeds:
                try:
                    generated = DataGenerator(
                        SimulationConfig(
                            n_samples=int(data_config.get("n_samples", 300)),
                            n_features=int(data_config.get("n_features", 10)),
                            random_state=int(seed),
                        )
                    ).generate(
                        data_config.get("mechanism", "weibull"),
                        c_mean=float(data_config.get("c_mean", 0.4)),
                    )
                    estimator = FederatedSurvival(
                        model=model,
                        protocol=protocol_name,
                        protocol_params=protocol_spec["params"],
                        n_clients=int(federated.get("n_clients", 5)),
                        global_rounds=int(federated.get("global_rounds", 2)),
                        local_steps=int(federated.get("local_steps", 1)),
                        batch_size=int(federated.get("batch_size", 32)),
                        learning_rate=float(federated.get("learning_rate", 1e-3)),
                        client_fraction=float(federated.get("client_fraction", 1.0)),
                        split_type=federated.get("split_type", "iid"),
                        split_alpha=float(federated.get("split_alpha", 0.5)),
                        random_state=int(seed),
                        num_nodes=(
                            () if model == "CoxPH" else tuple(federated.get("num_nodes", [32, 32]))
                        ),
                        num_durations=int(federated.get("num_durations", 25)),
                        optimizer=federated.get("optimizer", "adam"),
                        weight_decay=float(federated.get("weight_decay", 0.0)),
                    ).fit(generated)
                    summary = estimator.get_run_summary()
                    rows.append(
                        {
                            "protocol": protocol_name,
                            "model": model,
                            "seed": int(seed),
                            "method": protocol_name,
                            "c_index": summary["final_metrics"]["test_Cindex"],
                            "ibs": summary["final_metrics"]["test_IBS"],
                        }
                    )
                except Exception as error:
                    failures.append(
                        {
                            "protocol": protocol_name,
                            "model": model,
                            "seed": int(seed),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        recorder.write_metrics(metrics)
    failures_path = output_dir / "failures.csv"
    pd.DataFrame(
        failures,
        columns=["protocol", "model", "seed", "error_type", "error", "traceback"],
    ).to_csv(failures_path, index=False)
    recorder.add_artifact(failures_path, "failure_audit")
    recorder.finish("complete" if not failures else "partial")
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark cells failed; inspect {failures_path}")
    return output_dir


def run_config_file(config_path: Path, output_override: Path = None) -> Path:
    """Run either the schema-v1 workflow or a legacy simulation configuration.

    Schema-v1 files contain a singular ``model`` block and explicit ``data``,
    ``partition``, ``references``, ``evaluation``, and ``visualization``
    sections.  Existing multi-model smoke configurations remain supported.
    """
    config_path = Path(config_path).resolve()
    configuration = load_experiment_config(config_path)
    if is_workflow_configuration(configuration):
        return run_workflow_config(configuration, config_path, output_override)
    return _run_legacy_config(configuration, config_path, output_override)
