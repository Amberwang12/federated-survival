"""Regression tests for inert differential-privacy parameters.

The Gaussian mechanism scales its noise with ``dp_noise_multiplier`` and the
Laplace mechanism with ``dp_epsilon``.  Switching ``dp_mechanism`` therefore
turns the previously effective field into an inert one that still looks like a
working privacy control, and ``dp_delta`` never reaches the noise scale at all.
These tests pin both halves of the fix:

* the numerical half -- an inert field provably cannot change the noise, and
  the driver field provably does.  This is the property that actually matters;
* the reporting half -- :class:`FSAConfig` warns when an inert field has been
  set away from its default, and stays silent otherwise, and the two
  ``get_privacy_info`` implementations name the driver and the inert fields.
"""
import warnings

import numpy as np
import pytest
import torch

from federated_survival import DataGenerator, FederatedSurvival, SimulationConfig
from federated_survival.core.config import FSAConfig
from federated_survival.core.differential_privacy import (
    DP_MECHANISM_INERT_KNOBS,
    DP_MECHANISM_NOISE_KNOBS,
    DifferentialPrivacy,
    describe_noise_knobs,
)

MECHANISMS = ["gaussian", "laplace", "exponential"]


def _privatized(mechanism, seed=7, num_clients=3, **overrides):
    """Run one privatized update and return the resulting tensor.

    Both RNGs are reseeded before every call: the Gaussian path draws from
    torch and the Laplace path from numpy.  Two runs that differ only in a DP
    field therefore compare equal *only* when that field truly does not reach
    the noise.
    """
    config = FSAConfig(
        mode="simulate",
        num_clients=num_clients,
        use_differential_privacy=True,
        dp_mechanism=mechanism,
        **overrides,
    )
    tool = DifferentialPrivacy(config)
    global_weights = {"w": torch.zeros(4, 5)}
    local_weights = {"w": torch.full((4, 5), 0.3)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings():
        # The inert-parameter warning is asserted separately; it must not
        # obscure the numerical comparison here.
        warnings.simplefilter("ignore", UserWarning)
        return tool.privatize_model_update(
            global_weights, local_weights, num_clients
        )["w"]


def _make_config(**overrides):
    base = dict(
        mode="simulate",
        num_clients=3,
        use_differential_privacy=True,
        dp_mechanism="gaussian",
    )
    base.update(overrides)
    return FSAConfig(**base)


# --------------------------------------------------------------------------
# The core numerical claim
# --------------------------------------------------------------------------


def test_gaussian_dp_epsilon_does_not_change_noise():
    """dp_epsilon is inert under the Gaussian mechanism."""
    low = _privatized("gaussian", dp_epsilon=1.0)
    high = _privatized("gaussian", dp_epsilon=1000.0)
    assert torch.equal(low, high), (
        "dp_epsilon changed the Gaussian noise; if this now fails the mechanism "
        "gained an epsilon term and the inert-parameter warning is obsolete."
    )


def test_laplace_dp_noise_multiplier_does_not_change_noise():
    """dp_noise_multiplier is inert under the Laplace mechanism."""
    low = _privatized("laplace", dp_epsilon=10.0, dp_noise_multiplier=0.5)
    high = _privatized("laplace", dp_epsilon=10.0, dp_noise_multiplier=99.0)
    assert torch.equal(low, high)


@pytest.mark.parametrize("mechanism", ["gaussian", "laplace"])
def test_dp_delta_never_changes_noise(mechanism):
    """dp_delta does not reach the noise scale under either mechanism."""
    low = _privatized(mechanism, dp_delta=1e-9, dp_epsilon=10.0)
    high = _privatized(mechanism, dp_delta=1e-1, dp_epsilon=10.0)
    assert torch.equal(low, high)


def test_gaussian_dp_noise_multiplier_changes_noise():
    """The documented Gaussian driver really does drive the noise."""
    assert not torch.equal(
        _privatized("gaussian", dp_noise_multiplier=0.5),
        _privatized("gaussian", dp_noise_multiplier=5.0),
    )


def test_laplace_dp_epsilon_changes_noise():
    """The documented Laplace driver really does drive the noise."""
    assert not torch.equal(
        _privatized("laplace", dp_epsilon=10.0),
        _privatized("laplace", dp_epsilon=0.01),
    )


def test_switching_mechanism_changes_noise_for_the_same_epsilon():
    """The hazard being reported is real: the mechanism changes the noise.

    Holding ``dp_epsilon`` fixed and switching the mechanism changes the
    injected noise, which is exactly why a stale noise multiplier silently
    becomes meaningless.
    """
    gaussian = _privatized("gaussian", dp_epsilon=8.0, dp_noise_multiplier=1.0)
    laplace = _privatized("laplace", dp_epsilon=8.0, dp_noise_multiplier=1.0)
    assert not torch.equal(gaussian, laplace)


# --------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_noise_knob_mapping_covers_every_mechanism(mechanism):
    assert mechanism in DP_MECHANISM_NOISE_KNOBS
    assert mechanism in DP_MECHANISM_INERT_KNOBS


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_noise_knob_is_never_also_inert(mechanism):
    """The driver must not be listed as inert; that would be self-contradictory."""
    described = describe_noise_knobs(mechanism)
    assert described["noise_knob"] not in described["inert_knobs"]


def test_gaussian_and_laplace_drivers_are_different():
    """This difference is the whole reason the warning exists."""
    assert DP_MECHANISM_NOISE_KNOBS["gaussian"] == "dp_noise_multiplier"
    assert DP_MECHANISM_NOISE_KNOBS["laplace"] == "dp_epsilon"


def test_describe_noise_knobs_rejects_unknown_mechanism():
    with pytest.raises(ValueError, match="mechanism must be one of"):
        describe_noise_knobs("snapping_turtle")


@pytest.mark.parametrize("mechanism", ["gaussian", "laplace"])
def test_tool_accessors_agree_with_module_mapping(mechanism):
    config = _make_config(dp_mechanism=mechanism)
    tool = DifferentialPrivacy(config)
    assert tool.noise_driver() == DP_MECHANISM_NOISE_KNOBS[mechanism]
    assert tool.inactive_parameters() == DP_MECHANISM_INERT_KNOBS[mechanism]


# --------------------------------------------------------------------------
# The warning
# --------------------------------------------------------------------------


def _dp_warnings(**overrides):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_config(**overrides)
    return [
        str(item.message)
        for item in caught
        if "does not use" in str(item.message)
    ]


def test_warns_when_gaussian_user_tunes_epsilon():
    """The canonical mistake: tightening dp_epsilon under the Gaussian mechanism."""
    messages = _dp_warnings(dp_mechanism="gaussian", dp_epsilon=0.5)
    assert len(messages) == 1
    assert "dp_epsilon" in messages[0]
    assert "dp_noise_multiplier" in messages[0]


def test_warns_when_laplace_user_tunes_noise_multiplier():
    """The mirror-image mistake after switching to the Laplace mechanism."""
    messages = _dp_warnings(dp_mechanism="laplace", dp_noise_multiplier=0.2)
    assert len(messages) == 1
    assert "dp_noise_multiplier" in messages[0]
    assert "dp_epsilon" in messages[0]


def test_warns_for_inert_delta():
    messages = _dp_warnings(dp_mechanism="gaussian", dp_delta=1e-3)
    assert len(messages) == 1
    assert "dp_delta" in messages[0]


def test_warns_once_per_inert_field():
    messages = _dp_warnings(
        dp_mechanism="gaussian", dp_epsilon=0.5, dp_delta=1e-3
    )
    assert len(messages) == 2


def test_silent_when_inert_fields_are_left_at_their_defaults():
    """An untouched default cannot mislead anyone, so it must not warn."""
    assert _dp_warnings(dp_mechanism="gaussian") == []
    assert _dp_warnings(dp_mechanism="laplace") == []


@pytest.mark.parametrize(
    "mechanism,driver",
    [("gaussian", "dp_noise_multiplier"), ("laplace", "dp_epsilon")],
)
def test_silent_when_only_the_driver_is_tuned(mechanism, driver):
    messages = _dp_warnings(dp_mechanism=mechanism, **{driver: 0.25})
    assert messages == []


def test_silent_when_differential_privacy_is_disabled():
    """Without DP enabled there is no noise, so nothing is inert."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FSAConfig(
            mode="simulate",
            num_clients=3,
            dp_mechanism="laplace",
            dp_noise_multiplier=0.2,
        )
    assert [item for item in caught if "does not use" in str(item.message)] == []


def test_warning_names_the_offending_value():
    """The message must show what was ignored, not merely that it exists."""
    (message,) = _dp_warnings(dp_mechanism="gaussian", dp_epsilon=0.25)
    assert "0.25" in message


def test_warning_is_a_user_warning():
    """UserWarning keeps it visible without failing test runs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_config(dp_mechanism="gaussian", dp_epsilon=0.5)
    relevant = [item for item in caught if "does not use" in str(item.message)]
    assert relevant
    assert issubclass(relevant[0].category, UserWarning)


# --------------------------------------------------------------------------
# The reported metadata
# --------------------------------------------------------------------------


def test_runner_privacy_info_reports_driver_and_inert_fields():
    from federated_survival.core.runner import FSARunner

    config = _make_config(dp_mechanism="laplace", dp_epsilon=10.0)
    info = FSARunner(config).get_privacy_info()
    assert info["noise_driver"] == "dp_epsilon"
    assert "dp_noise_multiplier" in info["inactive_parameters"]
    assert info["formal_accounting_available"] is False


def test_api_privacy_info_reports_driver_and_inert_fields():
    estimator = FederatedSurvival(
        model="DeepSurv",
        n_clients=3,
        global_rounds=1,
        local_steps=1,
        random_state=7,
        use_differential_privacy=True,
        dp_mechanism="gaussian",
        dp_epsilon=8.0,
        dp_noise_multiplier=0.05,
    ).fit(_simulated_frame())
    info = estimator.get_privacy_info()
    assert info["noise_driver"] == "dp_noise_multiplier"
    assert "dp_epsilon" in info["inactive_parameters"]
    assert info["formal_accounting_available"] is False


def _simulated_frame(n=120, p=4):
    return DataGenerator(
        SimulationConfig(n_samples=n, n_features=p, random_state=13)
    ).generate("weibull", c_mean=0.4)


def test_privacy_info_disabled_path_is_unchanged():
    from federated_survival.core.runner import FSARunner

    info = FSARunner(_make_config(use_differential_privacy=False)).get_privacy_info()
    assert info == {"privacy_protection": False}
