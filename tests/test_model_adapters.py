import numpy as np
import pandas as pd
import pytest

from federated_survival.core import FSAConfig
from federated_survival.models import (
    ModelAdapter,
    PredictionValidationError,
    available_model_adapters,
    get_model_adapter,
    register_model_adapter,
    validate_survival_predictions,
)


BUILT_INS = {
    "CoxPH",
    "DeepSurv",
    "CoxCC",
    "CoxTime",
    "LogisticHazard",
    "PC-Hazard",
    "DeepHit",
}


def test_all_seven_builtin_adapters_are_registered():
    assert BUILT_INS.issubset(set(available_model_adapters()))
    assert all(get_model_adapter(name).name == name for name in BUILT_INS)


@pytest.mark.parametrize("model_name", sorted(BUILT_INS))
def test_builtin_adapter_constructs_model(model_name):
    kwargs = {"model_type": model_name, "n_features": 3, "num_durations": 8}
    if model_name == "CoxPH":
        kwargs["num_nodes"] = ()
    config = FSAConfig(**kwargs)
    adapter = get_model_adapter(model_name)
    durations = np.linspace(0.1, 4.0, 40)
    events = np.tile([0, 1], 20)
    label_transform = adapter.configure_targets(config, durations, events)
    network = adapter.build_network(config)
    model = adapter.build_model(network, config, label_transform)
    assert model.net is network


def test_custom_adapter_registration_extends_config_validation():
    class CustomAdapter(ModelAdapter):
        name = "UnitTestCustom"

        def build_model(self, network, config, label_transform=None, optimizer=None):
            return type("Wrapper", (), {"net": network})()

    register_model_adapter(CustomAdapter.name, CustomAdapter, replace=True)
    config = FSAConfig(model_type=CustomAdapter.name, n_features=2)
    assert get_model_adapter(config.model_type).name == CustomAdapter.name


def test_prediction_validation_reports_degenerate_but_valid_curves():
    survival = pd.DataFrame(
        [[1.0, 1.0], [0.8, 0.8], [0.5, 0.5]],
        index=[0.0, 1.0, 2.0],
    )
    diagnostics = validate_survival_predictions(survival)
    assert diagnostics.valid
    assert diagnostics.identical_across_samples


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 1.0], [np.nan, 0.8]],
        [[1.0, 1.0], [1.1, 0.8]],
        [[0.5, 0.5], [0.6, 0.4]],
    ],
)
def test_prediction_validation_rejects_invalid_curves(values):
    with pytest.raises(PredictionValidationError):
        validate_survival_predictions(pd.DataFrame(values))

