import json

import numpy as np
import pytest

from federated_survival import DataGenerator, FederatedSurvival, SimulationConfig
from federated_survival.models import available_model_adapters
from federated_survival.protocols import available_federated_protocols


def simulated_frame(n=120, p=4):
    return DataGenerator(SimulationConfig(n_samples=n, n_features=p, random_state=13)).generate(
        "weibull", c_mean=0.4
    )


def test_high_level_api_infers_features_and_predicts():
    frame = simulated_frame()
    estimator = FederatedSurvival(
        model="DeepSurv",
        n_clients=3,
        global_rounds=2,
        local_steps=1,
        random_state=7,
    ).fit(frame)
    assert estimator.config.n_features == 4
    assert estimator.config.num_clients == 3
    survival = estimator.predict_survival(frame.iloc[:5, :4])
    assert survival.shape[1] == 5
    assert np.isfinite(survival.to_numpy()).all()
    summary = estimator.get_run_summary()
    assert summary["model"] == "DeepSurv"
    assert estimator.get_privacy_info() == {
        "enabled": False,
        "formal_accounting_available": False,
    }


def test_high_level_api_reports_configured_dp_without_claiming_a_budget():
    estimator = FederatedSurvival(
        model="DeepSurv",
        n_clients=3,
        global_rounds=1,
        local_steps=1,
        random_state=7,
        use_differential_privacy=True,
        dp_mechanism="gaussian",
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_sensitivity=1.0,
        dp_noise_multiplier=0.05,
        dp_clip_norm=1.0,
    ).fit(simulated_frame())
    info = estimator.get_privacy_info()
    assert info["enabled"] is True
    assert info["mechanism"] == "gaussian"
    assert info["configured_noise_multiplier"] == 0.05
    assert info["formal_accounting_available"] is False
    assert "total_epsilon" not in info


def test_high_level_api_accepts_existing_client_column():
    frame = simulated_frame(n=180, p=3)
    frame["site"] = np.tile(["A", "B", "C"], 60)
    estimator = FederatedSurvival(
        model="PC-Hazard",
        n_clients=99,
        global_rounds=1,
        local_steps=1,
        num_durations=8,
    ).fit(frame, client_col="site")
    assert estimator.config.num_clients == 3
    assert estimator.config.n_features == 3


def test_high_level_api_accepts_numpy_arrays():
    frame = simulated_frame(n=100, p=2)
    estimator = FederatedSurvival(
        model="LogisticHazard",
        n_clients=2,
        global_rounds=1,
        num_durations=8,
    ).fit(
        frame[["x1", "x2"]].to_numpy(),
        frame["time"].to_numpy(),
        frame["status"].to_numpy(),
    )
    assert estimator.config.n_features == 2


def test_prediction_before_fit_has_actionable_error():
    estimator = FederatedSurvival(global_rounds=1)
    with pytest.raises(RuntimeError, match="fit must be called"):
        estimator.predict_survival(np.ones((2, 3)))


def test_available_models_and_protocols_expose_the_registries():
    estimator = FederatedSurvival(global_rounds=1)

    assert set(estimator.available_models) == set(available_model_adapters())
    assert set(estimator.available_protocols) == set(available_federated_protocols())
    assert {"CoxPH", "DeepSurv", "DeepHit"}.issubset(set(estimator.available_models))
    assert "FedAvg" in estimator.available_protocols


def _fitted_estimator(n=90, p=3, seed=5):
    frame = simulated_frame(n=n, p=p)
    estimator = FederatedSurvival(
        model="DeepSurv",
        n_clients=2,
        global_rounds=1,
        local_steps=1,
        num_durations=8,
        random_state=seed,
    ).fit(frame)
    return frame, estimator


def test_high_level_api_evaluates_a_fitted_model():
    frame, estimator = _fitted_estimator()
    features = frame[["x1", "x2", "x3"]].iloc[:25]

    report = estimator.evaluate(
        features,
        frame["time"].to_numpy()[:25],
        frame["status"].to_numpy()[:25],
    )

    assert set(report) == {"c_index", "ibs", "prediction_diagnostics"}
    assert 0.0 <= report["c_index"] <= 1.0
    assert report["ibs"] >= 0.0
    assert "prediction_diagnostics" in report
    json.dumps(report)  # 结果必须是可序列化的


def test_evaluate_rejects_mismatched_sample_counts():
    frame, estimator = _fitted_estimator()
    features = frame[["x1", "x2", "x3"]].iloc[:25]

    with pytest.raises(ValueError, match="same samples"):
        estimator.evaluate(
            features,
            frame["time"].to_numpy()[:10],
            frame["status"].to_numpy()[:10],
        )


def test_evaluate_requires_a_fitted_model():
    estimator = FederatedSurvival(global_rounds=1)
    with pytest.raises(RuntimeError, match="fit must be called"):
        estimator.evaluate(np.ones((3, 2)), np.ones(3), np.ones(3))
