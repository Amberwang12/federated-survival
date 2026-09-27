"""Protocol registry, optimization, and distributed-Cox regression tests."""

import numpy as np
import pandas as pd
import pytest
import torch
from lifelines import CoxPHFitter

from federated_survival import (
    DataGenerator,
    FederatedSurvival,
    SimulationConfig,
    available_federated_protocols,
    get_federated_protocol,
)
from federated_survival.core import FSAConfig
from federated_survival.protocols.parameter import ProximalLoss


BUILT_IN_PROTOCOLS = {"FedAvg", "FedProx", "FedOpt", "WebDISCO-style"}


def test_four_builtin_protocols_are_registered_with_aliases():
    assert BUILT_IN_PROTOCOLS.issubset(set(available_federated_protocols()))
    assert get_federated_protocol("fedadam").name == "FedOpt"
    assert get_federated_protocol("webdisco").name == "WebDISCO-style"


def test_webdisco_capability_validation_is_explicit():
    with pytest.raises(ValueError, match="supports only"):
        FSAConfig(model_type="DeepSurv", federated_protocol="WebDISCO-style")
    with pytest.raises(ValueError, match="full client participation"):
        FSAConfig(
            model_type="CoxPH",
            num_nodes=(),
            federated_protocol="WebDISCO-style",
            client_sample_ratio=0.5,
        )


def test_fedavg_returns_sample_weighted_parameters():
    protocol = get_federated_protocol("FedAvg")
    config = FSAConfig(model_type="DeepSurv", n_features=1)
    global_state = {"weight": torch.tensor([0.0])}
    protocol.begin_run(config, global_state)
    result = protocol.aggregate(
        global_state,
        [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([3.0])}],
        [0.25, 0.75],
        round_index=0,
    )
    assert torch.allclose(result["weight"], torch.tensor([2.5]))


def test_fedprox_loss_adds_distance_from_broadcast_model():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    base_loss = torch.nn.MSELoss()
    loss = ProximalLoss(
        base_loss,
        [("weight", parameter)],
        {"weight": torch.tensor([1.0])},
        mu=0.4,
    )
    value = loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert float(value) == pytest.approx(0.2)


def test_fedopt_keeps_server_state_and_moves_toward_client_average():
    protocol = get_federated_protocol("FedOpt")
    config = FSAConfig(
        model_type="DeepSurv",
        n_features=1,
        federated_protocol="FedOpt",
        server_learning_rate=0.1,
    )
    global_state = {"weight": torch.tensor([0.0])}
    protocol.begin_run(config, global_state)
    first = protocol.aggregate(
        global_state,
        [{"weight": torch.tensor([1.0])}, {"weight": torch.tensor([3.0])}],
        [0.5, 0.5],
        round_index=0,
    )
    second = protocol.aggregate(
        first,
        [{"weight": torch.tensor([2.0])}, {"weight": torch.tensor([2.0])}],
        [0.5, 0.5],
        round_index=1,
    )
    assert 0.0 < float(first["weight"]) < 2.0
    assert float(second["weight"]) > float(first["weight"])
    assert protocol.metadata()["server_optimizer"] == "adam"


def _small_frame():
    return DataGenerator(
        SimulationConfig(n_samples=140, n_features=3, random_state=17)
    ).generate("weibull", c_mean=0.4)


@pytest.mark.parametrize("protocol", ["FedAvg", "FedProx", "FedOpt"])
def test_parameter_protocols_run_through_high_level_api(protocol):
    estimator = FederatedSurvival(
        model="DeepSurv",
        protocol=protocol,
        n_clients=3,
        global_rounds=1,
        local_steps=2,
        batch_size=24,
        random_state=11,
        weight_decay=0.0,
    ).fit(_small_frame())
    summary = estimator.get_run_summary()
    assert summary["protocol"] == protocol
    assert np.isfinite(summary["final_metrics"]["test_Cindex"])
    assert np.isfinite(summary["final_metrics"]["test_IBS"])


def test_distributed_webdisco_statistics_equal_single_pooled_site():
    rng = np.random.RandomState(4)
    x = rng.normal(size=(30, 3))
    times = rng.uniform(0.1, 5.0, size=30)
    events = np.tile([1.0, 0.0, 1.0], 10)
    y = np.column_stack([times, events])
    clients = {"a": (x[:12], y[:12]), "b": (x[12:], y[12:])}
    pooled = {"pooled": (x, y)}
    beta = np.array([0.2, -0.1, 0.05])
    config = FSAConfig(
        model_type="CoxPH",
        num_nodes=(),
        n_features=3,
        federated_protocol="WebDISCO-style",
    )
    distributed = get_federated_protocol("WebDISCO-style")
    distributed.begin_run(config, {})
    event_times = distributed.event_times(clients)
    d_ll, d_score, d_info, _ = distributed.distributed_objective(
        clients, beta, event_times
    )
    p_ll, p_score, p_info, _ = distributed.distributed_objective(
        pooled, beta, event_times
    )
    assert d_ll == pytest.approx(p_ll, abs=1e-10)
    assert np.allclose(d_score, p_score, atol=1e-10)
    assert np.allclose(d_info, p_info, atol=1e-10)


def test_webdisco_runs_end_to_end_and_records_scope():
    estimator = FederatedSurvival(
        model="CoxPH",
        protocol="WebDISCO-style",
        n_clients=3,
        random_state=12,
        protocol_params={"max_iter": 20, "tolerance": 1e-7},
    ).fit(_small_frame())
    summary = estimator.get_run_summary()
    metadata = estimator.history_["protocol_metadata"]
    assert summary["protocol"] == "WebDISCO-style"
    assert np.isfinite(summary["final_metrics"]["test_Cindex"])
    assert np.isfinite(summary["final_metrics"]["test_IBS"])
    assert metadata["n_iterations"] >= 1
    assert metadata["payload_bytes_estimate"] > 0
    assert "not the original WebDISCO" in metadata["implementation_scope"]


def test_webdisco_coefficients_match_pooled_cox_on_continuous_times():
    frame = DataGenerator(
        SimulationConfig(n_samples=220, n_features=3, random_state=23)
    ).generate("weibull", c_mean=0.4)
    estimator = FederatedSurvival(
        model="CoxPH",
        protocol="WebDISCO-style",
        n_clients=4,
        random_state=23,
        protocol_params={"max_iter": 50, "tolerance": 1e-9},
    ).fit(frame)
    features = np.concatenate(
        [value[0] for value in estimator.dataset_.clients_set.values()]
    )
    labels = np.concatenate(
        [value[1] for value in estimator.dataset_.clients_set.values()]
    )
    pooled = pd.DataFrame(features, columns=["x1", "x2", "x3"])
    pooled["time"] = labels[:, 0]
    pooled["status"] = labels[:, 1]
    reference = CoxPHFitter().fit(
        pooled,
        duration_col="time",
        event_col="status",
        show_progress=False,
    )
    assert np.allclose(
        estimator.runner_.webdisco_fit_result_.coefficients,
        reference.params_.to_numpy(),
        atol=1e-6,
        rtol=1e-6,
    )
