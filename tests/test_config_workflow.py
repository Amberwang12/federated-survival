from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest
from federated_survival.cli import build_parser, command_config_options
from federated_survival.experiments.config_schema import normalize_workflow_config
from federated_survival.experiments.workflow_runner import run_workflow_config


def _write_survival_csv(path: Path) -> None:
    rows = []
    for index in range(36):
        rows.append(
            {
                "age": 40 + index,
                "marker": (index % 7) / 7,
                "follow_up": index + 1,
                "event": index % 2,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _configuration(data_name: str, output: Path, seeds=None):
    return {
        "schema_version": 1,
        "name": "configured-real-data-test",
        "experiment": {"seeds": seeds or [0, 1]},
        "data": {
            "source": "csv",
            "path": data_name,
            "duration_column": "follow_up",
            "event_column": "event",
            "test_size": 0.2,
            "standardize": True,
            "missing_values": "error",
        },
        "partition": {"method": "iid", "n_clients": 3, "stratify_by": "status"},
        "model": {"name": "DeepSurv", "hidden_nodes": [8], "dropout": 0.0},
        "federated": {
            "protocol": "FedAvg",
            "global_rounds": 2,
            "local_steps": 1,
            "batch_size": 8,
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
        "references": {
            "centralized": True,
            "local_models": True,
            "optimizer_steps": 2,
        },
        "evaluation": {
            "metrics": ["c_index", "ibs"],
            "save_round_metrics": True,
            "save_predictions": True,
        },
        "visualization": {
            "enabled": True,
            "formats": ["png"],
            "client_partition": {"type": "strip"},
            "round_metrics": {"type": "line", "aggregate_seeds": "mean_ci95"},
            "final_metrics": {
                "type": "boxplot",
                "metrics": ["c_index", "ibs"],
                "methods": ["Center", "Federated", "Weighted Local"],
            },
            "composite": {"enabled": True, "preset": "overview"},
        },
        "output": {"directory": str(output), "overwrite": False},
    }


def test_schema_resolves_real_input_relative_to_yaml(tmp_path):
    data = tmp_path / "survival.csv"
    _write_survival_csv(data)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    normalized = normalize_workflow_config(
        _configuration(data.name, tmp_path / "results"), config_path
    )
    assert normalized["data"]["resolved_path"] == str(data.resolve())
    assert normalized["experiment"]["seeds"] == [0, 1]
    assert normalized["visualization"]["final_metrics"]["type"] == "boxplot"


def test_boxplot_rejects_one_seed(tmp_path):
    data = tmp_path / "survival.csv"
    _write_survival_csv(data)
    config_path = tmp_path / "experiment.yaml"
    with pytest.raises(ValueError, match="requires at least two"):
        normalize_workflow_config(
            _configuration(data.name, tmp_path / "results", seeds=[0]), config_path
        )


def test_real_data_workflow_writes_metrics_audit_and_figures(tmp_path, monkeypatch):
    data = tmp_path / "survival.csv"
    _write_survival_csv(data)
    config_path = tmp_path / "experiment.yaml"
    output = tmp_path / "results"
    configuration = _configuration(data.name, output)

    def fake_baselines(
        config,
        dataset,
        seed,
        baseline_epochs=None,
        telemetry_rows=None,
        prediction_rows=None,
        include_center=True,
        include_local=True,
    ):
        if telemetry_rows is not None:
            for round_index in (1, 2):
                telemetry_rows.append(
                    {
                        "seed": seed,
                        "model": config.model_type,
                        "round": round_index,
                        "train_c_index": 0.60 + seed * 0.01,
                        "train_ibs": 0.20,
                        "test_c_index": 0.58 + seed * 0.01 + round_index * 0.01,
                        "test_ibs": 0.22 - round_index * 0.01,
                        "train_loss": 1.0,
                        "update_direction_norm": 1.0,
                        "client_drift": 0.1,
                        "update_direction_dispersion": 0.2,
                        "communication_bytes": 100,
                        "selected_clients": "[]",
                    }
                )
        if prediction_rows is not None:
            prediction_rows.append(
                {
                    "seed": seed,
                    "model": config.model_type,
                    "method": "FSA",
                    "sample_id": 0,
                    "time": 1.0,
                    "survival_probability": 0.9,
                }
            )
        rows = [
            ("Center", 0.62 + seed * 0.01, 0.19),
            ("FSA", 0.60 + seed * 0.01, 0.20),
            ("Local", 0.57 + seed * 0.01, 0.22),
        ]
        return pd.DataFrame(
            [
                {
                    "seed": seed,
                    "model": config.model_type,
                    "method": method,
                    "client": "all",
                    "n_train": len(dataset.train_data),
                    "c_index": c_index,
                    "ibs": ibs,
                }
                for method, c_index, ibs in rows
            ]
        )

    monkeypatch.setattr(
        "federated_survival.experiments.workflow_runner.run_paired_baselines",
        fake_baselines,
    )
    result = run_workflow_config(configuration, config_path)
    assert result == output
    for relative in (
        "resolved_config.yaml",
        "environment.json",
        "data_summary.json",
        "client_partition.csv",
        "client_partition_observations.csv",
        "raw_results.csv",
        "summary_metrics.csv",
        "round_metrics.csv",
        "survival_predictions.csv",
        "failures.csv",
        "run_status.json",
        "run_manifest.json",
        "figures/client_partition.png",
        "figures/round_metrics.png",
        "figures/final_metrics_boxplot.png",
        "figures/overview.png",
    ):
        assert (output / relative).exists(), relative
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status["complete"] is True
    assert status["complete_seeds"] == 2
    assert status["failed_seeds"] == 0


def test_cli_exposes_config_options():
    parsed = build_parser().parse_args(["config-options"])
    assert parsed.command == "config-options"


def test_data_only_simulation_writes_partition_and_scatter_without_training(tmp_path):
    config_path = tmp_path / "simulation.yaml"
    output = tmp_path / "results"
    configuration = {
        "schema_version": 1,
        "name": "simulation-preview-test",
        "experiment": {"stage": "data_only", "seeds": [42], "fail_fast": True},
        "data": {
            "source": "simulation",
            "mechanism": "weibull",
            "n_samples": 60,
            "n_features": 3,
            "censoring_rate": 0.4,
            "test_size": 0.2,
            "standardize": False,
        },
        "partition": {"method": "iid", "n_clients": 3},
        "visualization": {
            "enabled": True,
            "formats": ["png"],
            "client_partition": {
                "type": "scatter",
                "x": "feature_1",
                "y": "time",
            },
            "composite": {"enabled": False},
        },
        "output": {"directory": str(output), "overwrite": False},
    }

    result = run_workflow_config(configuration, config_path)

    assert result == output
    for relative in (
        "simulated_data.csv",
        "client_partition.csv",
        "client_partition_observations.csv",
        "figures/client_partition.png",
        "run_status.json",
        "run_manifest.json",
    ):
        assert (output / relative).exists(), relative
    assert not (output / "raw_results.csv").exists()
    observations = pd.read_csv(output / "client_partition_observations.csv")
    assert {"feature_1", "feature_2", "feature_3"}.issubset(observations.columns)
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert status["complete"] is True
    assert status["complete_seeds"] == 1


def test_composite_overview_is_dataset_independent(tmp_path):
    data = tmp_path / "survival.csv"
    _write_survival_csv(data)
    configuration = _configuration(data.name, tmp_path / "results", seeds=[0])
    configuration["visualization"]["final_metrics"]["type"] = "dotplot"
    normalized = normalize_workflow_config(configuration, tmp_path / "experiment.yaml")
    assert normalized["visualization"]["composite"]["preset"] == "overview"


def test_config_options_have_no_dataset_specific_entrypoints(capsys):
    command_config_options(Namespace(output=None))
    output = json.loads(capsys.readouterr().out)
    assert "paper_example" not in output
    assert "multiseed_example" not in output


def test_cli_has_no_chapter3_reproduction_entrypoint():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    assert "reproduce-paper" not in subparsers.choices
