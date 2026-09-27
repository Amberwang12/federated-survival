"""Coverage for the published ``federated-survival`` console command.

``pyproject.toml`` exposes ``federated-survival = federated_survival.cli:main``
as the only entry point a user touches after installing the package.  These
tests drive ``main`` in-process (it accepts an argument list) so the
sub-command dispatch, every command body, and the configuration runner behind
``run`` are all exercised without spawning a subprocess.

The single genuinely expensive step -- a real federated quickstart -- is run
once with a deliberately tiny budget.  Everything else stubs the estimator so
the suite stays fast.
"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from federated_survival.cli import (
    build_parser,
    command_config_options,
    command_doctor,
    command_quickstart,
    doctor_report,
    main,
)
from federated_survival.experiments.config_runner import (
    _protocol_specs,
    _run_legacy_config,
    run_config_file,
)

MINIMAL_QUICKSTART = [
    "quickstart",
    "--rounds",
    "1",
    "--samples",
    "60",
    "--features",
    "3",
    "--clients",
    "2",
    "--num-durations",
    "5",
    "--split-type",
    "random",
]


class _StubEstimator:
    """Record constructor arguments and return a fixed, valid metric row."""

    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataset_ = None
        type(self).calls.append(kwargs)

    def fit(self, frame, *args, **kwargs):
        return self

    def get_run_summary(self):
        return {
            "model": self.kwargs.get("model", "DeepSurv"),
            "final_metrics": {"test_Cindex": 0.61, "test_IBS": 0.19},
        }


@pytest.fixture(autouse=True)
def _reset_stub_calls():
    _StubEstimator.calls = []
    yield
    _StubEstimator.calls = []


def _legacy_configuration(output: Path, **overrides):
    configuration = {
        "name": "cli-legacy",
        "data": {"n_samples": 60, "n_features": 3, "mechanism": "weibull", "c_mean": 0.4},
        "federated": {"n_clients": 2, "global_rounds": 1, "local_steps": 1, "num_durations": 5},
        "models": ["DeepSurv"],
        "seeds": [1],
        "output": str(output),
    }
    configuration.update(overrides)
    return configuration


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Sub-command dispatch
# --------------------------------------------------------------------------


def test_main_requires_a_subcommand():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_main_rejects_an_unknown_subcommand():
    with pytest.raises(SystemExit) as error:
        main(["definitely-not-a-command"])
    assert error.value.code == 2


def test_parser_exposes_the_documented_commands():
    parser = build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparsers.choices) == {"doctor", "quickstart", "run", "config-options"}


def test_main_dispatches_doctor(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["package_version"]


def test_main_dispatches_config_options(capsys):
    assert main(["config-options"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "models" in report and "protocols" in report


def test_main_runs_quickstart_end_to_end(tmp_path, capsys):
    output = tmp_path / "quickstart"

    exit_code = main(MINIMAL_QUICKSTART + ["--output", str(output)])

    assert exit_code == 0
    for name in (
        "round_metrics.csv",
        "survival_predictions.csv",
        "summary.json",
        "run_manifest.json",
    ):
        assert (output / name).exists(), name

    metrics = pd.read_csv(output / "round_metrics.csv")
    assert list(metrics.columns) == [
        "round",
        "train_Cindex",
        "train_IBS",
        "test_Cindex",
        "test_IBS",
    ]
    assert len(metrics) == 1

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "DeepSurv"
    assert "Quickstart completed" in capsys.readouterr().out


def test_main_dispatches_run_to_the_config_runner(tmp_path, monkeypatch, capsys):
    config_path = _write_json(tmp_path / "legacy.json", _legacy_configuration(tmp_path / "out"))
    resolved = tmp_path / "resolved-output"
    captured = {}

    def fake_run_config_file(path, output_override=None):
        captured["path"] = path
        captured["output_override"] = output_override
        return resolved

    monkeypatch.setattr("federated_survival.cli.run_config_file", fake_run_config_file)

    exit_code = main(["run", "--config", str(config_path), "--output", str(resolved)])

    assert exit_code == 0
    assert captured["path"] == Path(config_path)
    assert captured["output_override"] == resolved
    assert str(resolved.resolve()) in capsys.readouterr().out


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_report_summarizes_the_registry():
    report = doctor_report()
    assert report["status"] == "passed"
    assert all(report["adapters"].values())
    assert all(report["protocols"].values())
    assert "environment" in report


def test_doctor_report_marks_a_broken_adapter_as_failed(monkeypatch):
    def explode(name):
        raise RuntimeError("adapter exploded")

    monkeypatch.setattr("federated_survival.cli.get_model_adapter", explode)
    report = doctor_report()

    assert report["status"] == "failed"
    assert report["adapters"] and not any(report["adapters"].values())


def test_doctor_report_marks_a_broken_protocol_as_failed(monkeypatch):
    def explode(name):
        raise RuntimeError("protocol exploded")

    monkeypatch.setattr("federated_survival.cli.get_federated_protocol", explode)
    report = doctor_report()

    assert report["status"] == "failed"
    assert report["protocols"] and not any(report["protocols"].values())


def test_doctor_writes_an_optional_report_file(tmp_path, capsys):
    destination = tmp_path / "doctor.json"
    exit_code = command_doctor(Namespace(output=str(destination)))

    assert exit_code == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == "passed"
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


# --------------------------------------------------------------------------
# config-options
# --------------------------------------------------------------------------


def test_config_options_writes_an_optional_report_file(tmp_path, capsys):
    destination = tmp_path / "options.json"
    assert command_config_options(Namespace(output=str(destination))) == 0

    written = json.loads(destination.read_text(encoding="utf-8"))
    assert "CoxPH" in written["models"]
    assert "FedAvg" in written["protocols"]
    assert json.loads(capsys.readouterr().out)["documentation"]


# --------------------------------------------------------------------------
# quickstart failure handling
# --------------------------------------------------------------------------


def test_quickstart_records_failure_and_reraises(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("simulation exploded")

    monkeypatch.setattr("federated_survival.cli.DataGenerator", explode)
    args = build_parser().parse_args(["quickstart", "--output", str(tmp_path)])

    with pytest.raises(RuntimeError, match="simulation exploded"):
        command_quickstart(args)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "simulation exploded" in manifest["error"]


# --------------------------------------------------------------------------
# _protocol_specs
# --------------------------------------------------------------------------


def test_protocol_specs_default_to_the_federated_block():
    specs = _protocol_specs({}, {"protocol": "FedProx", "protocol_params": {"mu": 0.2}})
    assert specs == [{"name": "FedProx", "params": {"mu": 0.2}, "models": None}]


def test_protocol_specs_default_protocol_name():
    assert _protocol_specs({}, {})[0]["name"] == "FedAvg"


def test_protocol_specs_accept_string_and_mapping_entries():
    specs = _protocol_specs(
        {"protocols": ["FedAvg", {"name": "FedProx", "params": {"mu": 0.1}, "models": ["DeepSurv"]}]},
        {},
    )
    assert specs == [
        {"name": "FedAvg", "params": {}, "models": None},
        {"name": "FedProx", "params": {"mu": 0.1}, "models": ["DeepSurv"]},
    ]


def test_protocol_specs_reject_an_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        _protocol_specs({"protocols": []}, {})


def test_protocol_specs_reject_a_non_list():
    with pytest.raises(ValueError, match="non-empty list"):
        _protocol_specs({"protocols": "FedAvg"}, {})


def test_protocol_specs_reject_an_entry_without_a_name():
    with pytest.raises(ValueError, match="name or an object"):
        _protocol_specs({"protocols": [{"params": {}}]}, {})


def test_protocol_specs_reject_empty_models():
    with pytest.raises(ValueError, match="non-empty list when provided"):
        _protocol_specs({"protocols": [{"name": "FedAvg", "models": []}]}, {})


# --------------------------------------------------------------------------
# _run_legacy_config
# --------------------------------------------------------------------------


def test_legacy_config_rejects_an_empty_model_list(tmp_path):
    configuration = _legacy_configuration(tmp_path / "out", models=[])
    with pytest.raises(ValueError, match="models must be a non-empty list"):
        _run_legacy_config(configuration, tmp_path / "legacy.json")


def test_legacy_config_rejects_an_empty_seed_list(tmp_path):
    configuration = _legacy_configuration(tmp_path / "out", seeds=[])
    with pytest.raises(ValueError, match="seeds must be a non-empty list"):
        _run_legacy_config(configuration, tmp_path / "legacy.json")


def test_legacy_config_rejects_protocol_models_outside_the_selection(tmp_path):
    configuration = _legacy_configuration(
        tmp_path / "out",
        protocols=[{"name": "FedAvg", "models": ["DeepHit"]}],
    )
    with pytest.raises(ValueError, match="names models not present"):
        _run_legacy_config(configuration, tmp_path / "legacy.json")


def test_legacy_config_writes_metrics_for_stubbed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "federated_survival.experiments.config_runner.FederatedSurvival",
        _StubEstimator,
    )
    output = tmp_path / "out"
    configuration = _legacy_configuration(output, seeds=[1, 2])

    result = _run_legacy_config(configuration, tmp_path / "legacy.json")

    assert result == output
    metrics = pd.read_csv(output / "metrics.csv")
    assert list(metrics.columns) == ["protocol", "model", "seed", "method", "c_index", "ibs"]
    assert sorted(metrics["seed"]) == [1, 2]

    failures = pd.read_csv(output / "failures.csv")
    assert failures.empty

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert (output / "resolved_config.json").exists()


def test_legacy_config_records_failed_cells(tmp_path, monkeypatch):
    class _ExplodingEstimator(_StubEstimator):
        def fit(self, frame, *args, **kwargs):
            raise RuntimeError("cell failed")

    monkeypatch.setattr(
        "federated_survival.experiments.config_runner.FederatedSurvival",
        _ExplodingEstimator,
    )
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="benchmark cells failed"):
        _run_legacy_config(_legacy_configuration(output), tmp_path / "legacy.json")

    failures = pd.read_csv(output / "failures.csv")
    assert len(failures) == 1
    assert failures.iloc[0]["error_type"] == "RuntimeError"
    assert "cell failed" in failures.iloc[0]["traceback"]

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"


# --------------------------------------------------------------------------
# run_config_file dispatch
# --------------------------------------------------------------------------


def _workflow_configuration(output: Path):
    return {
        "schema_version": 1,
        "name": "cli-workflow",
        "experiment": {"stage": "data_only", "seeds": [42], "fail_fast": True},
        "data": {
            "source": "simulation",
            "mechanism": "weibull",
            "n_samples": 40,
            "n_features": 3,
            "test_size": 0.2,
            "standardize": False,
        },
        "partition": {"method": "iid", "n_clients": 2},
        "visualization": {"enabled": False, "composite": {"enabled": False}},
        "output": {"directory": str(output), "overwrite": False},
    }


def test_run_config_file_dispatches_the_workflow_schema(tmp_path):
    output = tmp_path / "workflow-out"
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        json.dumps(_workflow_configuration(output)), encoding="utf-8"
    )

    result = run_config_file(config_path)

    assert result == output
    assert (output / "simulated_data.csv").exists()


def test_run_config_file_dispatches_the_legacy_schema(tmp_path, monkeypatch):
    config_path = _write_json(tmp_path / "legacy.json", _legacy_configuration(tmp_path / "out"))
    sentinel = tmp_path / "legacy-out"
    captured = {}

    def fake_legacy(configuration, path, output_override=None):
        captured["configuration"] = configuration
        captured["path"] = path
        return sentinel

    monkeypatch.setattr(
        "federated_survival.experiments.config_runner._run_legacy_config", fake_legacy
    )

    assert run_config_file(config_path) == sentinel
    assert captured["configuration"]["name"] == "cli-legacy"


def test_run_config_file_reads_yaml_configurations(tmp_path, monkeypatch):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        "name: cli-legacy\ndata:\n  n_samples: 40\nmodels: [DeepSurv]\nseeds: [1]\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "out"
    monkeypatch.setattr(
        "federated_survival.experiments.config_runner._run_legacy_config",
        lambda configuration, path, output_override=None: sentinel,
    )

    assert run_config_file(config_path) == sentinel


def test_run_config_file_rejects_an_unknown_extension(tmp_path):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text("name = 'x'", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must use \.json, \.yaml, or \.yml"):
        run_config_file(config_path)


def test_run_config_file_rejects_a_non_mapping_top_level(tmp_path):
    config_path = _write_json(tmp_path / "list.json", [1, 2, 3])

    with pytest.raises(ValueError, match="mapping at the top level"):
        run_config_file(config_path)
