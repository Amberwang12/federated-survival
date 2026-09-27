import json

import numpy as np
import pandas as pd
import pytest

from federated_survival.cli import build_parser, doctor_report
from federated_survival.experiments.infrastructure import (
    ExperimentRecorder,
    load_experiment_config,
    validate_metrics,
)


def test_recorder_writes_manifest_and_hashed_metrics(tmp_path):
    recorder = ExperimentRecorder(tmp_path, "unit-test", {"seed": 42})
    metrics = pd.DataFrame({"c_index": [0.7], "ibs": [0.2]})
    output = recorder.write_metrics(metrics)
    recorder.finish()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["configuration"] == {"seed": 42}
    assert manifest["artifacts"][0]["path"] == str(output.resolve())
    assert len(manifest["artifacts"][0]["sha256"]) == 64


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"value": [1.0]}),
        pd.DataFrame({"c_index": [np.nan]}),
        pd.DataFrame({"c_index": [1.1]}),
        pd.DataFrame({"ibs": [-0.1]}),
    ],
)
def test_metric_validation_rejects_invalid_exports(frame):
    with pytest.raises(ValueError):
        validate_metrics(frame)


def test_load_json_experiment_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"models": ["DeepSurv"], "seeds": [1]}')
    assert load_experiment_config(path)["models"] == ["DeepSurv"]


def test_doctor_checks_all_seven_adapters():
    report = doctor_report()
    assert report["status"] == "passed"
    assert len(report["adapters"]) >= 7
    assert all(report["adapters"].values())


def test_cli_has_reproducibility_commands():
    parser = build_parser()
    for command in ("doctor", "quickstart", "run"):
        parsed = parser.parse_args([command] + ({
            "run": ["--config", "config.json"],
        }.get(command, [])))
        assert parsed.command == command
