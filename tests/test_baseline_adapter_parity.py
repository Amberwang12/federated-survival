"""Parity between the centralized-baseline helpers and the adapter registry.

``experiments.baselines`` originally resolved label transforms and model
wrappers through hard-coded ``if/elif`` chains over ``config.model_type``,
while ``core.runner`` resolves both through the public adapter registry.  The
two agreed for the seven built-in models but diverged for any adapter
registered through the documented ``register_model_adapter`` extension point:
the baseline path silently produced ``out_features = 1`` instead of the
adapter's real output width, and ``_wrap_model`` refused the model outright.

These tests pin both implementations to the same answers.  The built-in cases
lock in the numbers used by the paper's comparison table; the custom-adapter
cases would have caught the divergence before it reached a publication.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pycox.models import DeepHitSingle

from federated_survival.core.config import FSAConfig
from federated_survival.experiments.baselines import _fit_one, _prepare, _wrap_model
from federated_survival.models import (
    ModelAdapter,
    available_model_adapters,
    get_model_adapter,
)
from federated_survival.models import adapters as adapters_module

BUILTIN_MODELS = (
    "CoxPH",
    "DeepSurv",
    "CoxCC",
    "CoxTime",
    "LogisticHazard",
    "PC-Hazard",
    "DeepHit",
)

CUSTOM_MODEL = "ParityCustomDeepHit"
_CUSTOM_KEY = CUSTOM_MODEL.lower()


class _CustomDeepHitAdapter(ModelAdapter):
    """A third-party DeepHit variant, registered the documented way."""

    name = CUSTOM_MODEL
    uses_label_transform = True

    def create_label_transform(self, config):
        return DeepHitSingle.label_transform(config.num_durations, scheme="quantiles")

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return DeepHitSingle(network, optimizer, duration_index=label_transform.cuts)


@pytest.fixture
def custom_model(monkeypatch):
    """Register the custom adapter for one test, then restore the registry."""
    monkeypatch.setitem(adapters_module._REGISTRY, _CUSTOM_KEY, _CustomDeepHitAdapter)
    monkeypatch.setitem(adapters_module._CANONICAL_NAMES, _CUSTOM_KEY, CUSTOM_MODEL)
    return CUSTOM_MODEL


def _synthetic_targets(n_samples=200, seed=0):
    rng = np.random.RandomState(seed)
    durations = rng.exponential(2.0, n_samples)
    events = rng.randint(0, 2, n_samples)
    return np.column_stack([durations, events]).astype("float64")


def _config_kwargs(model_type):
    kwargs = {"model_type": model_type, "num_durations": 10}
    if model_type == "CoxPH":
        # CoxPH has no hidden layers, so the architecture must be empty.
        kwargs["num_nodes"] = ()
    return kwargs


def _as_arrays(value):
    if isinstance(value, tuple):
        return tuple(np.asarray(part, dtype=float) for part in value)
    return np.asarray(value, dtype=float)


def _assert_same_targets(left, right):
    left_arrays = _as_arrays(left)
    right_arrays = _as_arrays(right)
    if isinstance(left_arrays, tuple):
        assert len(left_arrays) == len(right_arrays)
        for lhs, rhs in zip(left_arrays, right_arrays):
            np.testing.assert_allclose(lhs, rhs)
    else:
        np.testing.assert_allclose(left_arrays, right_arrays)


@pytest.mark.parametrize("model", BUILTIN_MODELS)
def test_builtin_models_agree_on_target_configuration(model):
    """The two implementations must report the same width and transform type."""
    targets = _synthetic_targets()

    baseline_config = FSAConfig(**_config_kwargs(model))
    baseline_transform, baseline_targets = _prepare(baseline_config, targets)

    adapter = get_model_adapter(model)
    adapter_config = FSAConfig(**_config_kwargs(model))
    adapter_transform = adapter.configure_targets(
        adapter_config, targets[:, 0], targets[:, 1]
    )
    adapter_targets = adapter.transform_target(targets, adapter_transform)

    assert baseline_config.out_features == adapter_config.out_features
    assert type(baseline_transform) is type(adapter_transform)
    _assert_same_targets(baseline_targets, adapter_targets)


@pytest.mark.parametrize("model", BUILTIN_MODELS)
def test_builtin_models_agree_on_wrapped_model_type(model):
    """Wrapping must produce the same pycox class through either route."""
    config = FSAConfig(**_config_kwargs(model))
    label_transform, _ = _prepare(config, _synthetic_targets())

    network = get_model_adapter(model).build_network(config)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    from_baselines = _wrap_model(config, network, optimizer, label_transform)
    from_adapter = get_model_adapter(model).build_model(
        network, config, label_transform=label_transform, optimizer=optimizer
    )

    assert type(from_baselines) is type(from_adapter)
    # pycox wraps the optimizer, so unwrap once before comparing identity.
    assert from_baselines.optimizer.optimizer is optimizer
    assert from_adapter.optimizer.optimizer is optimizer


def test_registry_extension_point_reaches_config(custom_model):
    """A registered adapter is a supported FSAConfig value, not a special case."""
    config = FSAConfig(model_type=custom_model, num_durations=10)
    assert config.model_type == custom_model
    assert custom_model in available_model_adapters()


def test_custom_adapter_agrees_on_target_configuration(custom_model):
    """Baselines must not fall back to a width of 1 for a custom adapter."""
    targets = _synthetic_targets()

    adapter = get_model_adapter(custom_model)
    adapter_config = FSAConfig(model_type=custom_model, num_durations=10)
    adapter_transform = adapter.configure_targets(
        adapter_config, targets[:, 0], targets[:, 1]
    )

    baseline_config = FSAConfig(model_type=custom_model, num_durations=10)
    baseline_transform, baseline_targets = _prepare(baseline_config, targets)

    assert baseline_config.out_features == adapter_config.out_features
    assert type(baseline_transform) is type(adapter_transform)
    _assert_same_targets(
        baseline_targets, adapter.transform_target(targets, adapter_transform)
    )


def test_custom_adapter_can_be_wrapped_by_baseline_helper(custom_model):
    """``_wrap_model`` must defer to the adapter instead of raising."""
    config = FSAConfig(model_type=custom_model, num_durations=10)
    label_transform, _ = _prepare(config, _synthetic_targets())
    network = get_model_adapter(custom_model).build_network(config)

    wrapped = _wrap_model(config, network, None, label_transform)
    assert type(wrapped) is DeepHitSingle


def test_baseline_hazard_requirement_follows_adapter_declaration():
    """``_fit_one`` asks the adapter which models need baseline hazards.

    The declaration replaces a hard-coded ``model_type`` list, so this test
    pins the set the baselines actually rely on.
    """
    declared = {
        name for name in BUILTIN_MODELS if get_model_adapter(name).requires_baseline_hazards
    }
    assert declared == {"CoxPH", "DeepSurv", "CoxCC", "CoxTime"}


def test_custom_adapter_runs_through_centralized_baseline(custom_model):
    """End-to-end proof that a custom adapter works on the baseline path."""
    rng = np.random.RandomState(5)
    n_samples = 90
    features = rng.randn(n_samples, 3).astype("float32")
    targets = _synthetic_targets(n_samples, seed=5)

    config = FSAConfig(
        model_type=custom_model,
        n_features=3,
        num_durations=5,
        num_clients=2,
        batch_size=16,
    )
    c_index, ibs = _fit_one(
        config,
        features[:70],
        targets[:70],
        features[70:],
        targets[70:],
        seed=0,
        epochs=2,
    )
    assert np.isfinite(c_index)
    assert np.isfinite(ibs)
