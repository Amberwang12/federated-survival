"""Regression tests for numerically stable Cox survival predictions.

``pycox`` evaluates the cumulative hazard as ``H0(t) * exp(logh)``.  With a
large log-hazard -- which happens when differential-privacy noise swamps the
weights, or when a fit diverges -- ``exp(logh)`` becomes ``inf`` and the
product collapses to ``inf * 0 = nan`` at every time whose baseline
cumulative hazard is zero.  ``stable_cox_survival`` must return a finite,
bounded, non-increasing matrix in that situation instead.
"""

import numpy as np
import pandas as pd
import pytest

from federated_survival.models.adapters import (
    _LOG_CUMHAZ_CAP,
    stable_cox_survival,
)


class _StubCox:
    """Minimal stand-in exposing the two attributes ``stable_cox_survival`` reads."""

    def __init__(self, log_hazard, baseline_cumhaz):
        self._log_hazard = np.asarray(log_hazard, dtype=float)
        self.baseline_cumulative_hazards_ = pd.Series(
            np.asarray(baseline_cumhaz, dtype=float),
            index=np.arange(len(baseline_cumhaz), dtype=float),
        )

    def predict(self, features):
        return self._log_hazard[: len(features)]


def _pycox_reference(log_hazard, baseline_cumhaz):
    """The unfixed ``exp(-H0 * exp(logh))`` computation, for comparison."""
    expg = np.exp(np.asarray(log_hazard, dtype=float)).reshape(1, -1)
    h0 = np.asarray(baseline_cumhaz, dtype=float).reshape(-1, 1)
    return np.exp(-h0.dot(expg))


def test_reference_path_nan_when_hazard_overflows():
    """Guard the bug itself: the naive path is non-finite, so the fix is meaningful.

    ``np.exp`` overflows past ~709, and ``inf * 0`` is ``nan`` -- here at the
    time point whose baseline cumulative hazard is zero.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        reference = _pycox_reference([800.0, 801.0, 802.0], [0.0, 0.1, 0.3, 0.5])
    assert not np.isfinite(reference).all()


def test_stable_path_is_finite_for_large_log_hazard():
    model = _StubCox([800.0, 801.0, 802.0], [0.0, 0.1, 0.3, 0.5])
    survival = stable_cox_survival(model, np.zeros((3, 1)))
    values = survival.to_numpy(dtype=float)

    assert values.shape == (4, 3)
    assert np.isfinite(values).all()


def test_stable_path_stays_within_probability_bounds():
    model = _StubCox([500.0, -500.0, 0.0], [0.0, 0.1, 0.3, 0.5])
    values = stable_cox_survival(model, np.zeros((3, 1))).to_numpy(dtype=float)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


def test_stable_path_is_monotone_non_increasing_in_time():
    model = _StubCox([400.0, 401.0], [0.0, 0.05, 0.2, 0.6, 1.5])
    values = stable_cox_survival(model, np.zeros((2, 1))).to_numpy(dtype=float)
    assert np.all(np.diff(values, axis=0) <= 1e-12)


def test_zero_baseline_cumhaz_keeps_full_survival():
    """Before the first event time, S(t|x) must stay 1 even for a huge hazard."""
    model = _StubCox([800.0], [0.0, 0.2, 0.4])
    values = stable_cox_survival(model, np.zeros((1, 1))).to_numpy(dtype=float)
    assert values[0, 0] == pytest.approx(1.0)


def test_saturated_cumhaz_gives_zero_survival():
    """A hazard far beyond the cap must yield an exact 0, never a negative value."""
    model = _StubCox([_LOG_CUMHAZ_CAP * 10], [1.0])
    values = stable_cox_survival(model, np.zeros((1, 1))).to_numpy(dtype=float)
    assert values[0, 0] == pytest.approx(0.0)


def test_stable_path_matches_reference_in_well_conditioned_regime():
    """Away from overflow the stable formula must reproduce the original values."""
    log_hazard = [-0.5, 0.0, 0.5]
    baseline = [0.0, 0.1, 0.3, 0.5]
    model = _StubCox(log_hazard, baseline)

    values = stable_cox_survival(model, np.zeros((3, 1))).to_numpy(dtype=float)
    reference = _pycox_reference(log_hazard, baseline)
    assert np.allclose(values, reference, atol=1e-12)


def test_index_is_preserved_from_baseline_hazards():
    model = _StubCox([0.1], [0.0, 0.2, 0.4])
    survival = stable_cox_survival(model, np.zeros((1, 1)))
    assert list(survival.index) == list(model.baseline_cumulative_hazards_.index)
