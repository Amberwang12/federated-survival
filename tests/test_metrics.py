"""Regression tests for the documented public metric helpers.

These cover ``federated_survival.utils.metrics``, which is exported at package
level (``fs.calculate_cindex`` / ``fs.calculate_ibs``) but was previously
untested.  The suite is expected to pass on both NumPy < 2.0 and NumPy >= 2.0.
"""

import numpy as np
import pytest

from federated_survival.utils.metrics import (
    calculate_cindex,
    calculate_ibs,
    evaluation_time_grid,
)


def _toy_data(n=120, seed=0):
    rng = np.random.RandomState(seed)
    time = rng.exponential(2.0, n)
    event = (rng.uniform(size=n) > 0.3).astype(float)
    return time, event


def _constant_curve(grid, rate, n):
    return np.tile(np.exp(-rate * grid)[:, None], (1, n))


# --------------------------------------------------------------- C-index

def test_cindex_is_one_for_perfect_ranking():
    """``risk_score`` is a hazard-style score: higher means earlier event.

    ``calculate_cindex`` negates the score before handing it to
    ``lifelines.concordance_index`` (which expects a predicted survival
    time), so a correctly ranked model has risk decreasing with time.
    """
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    assert calculate_cindex(time, event, risk) == pytest.approx(1.0)


def test_cindex_is_zero_for_inverted_ranking():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)
    risk = np.array([1.0, 2.0, 3.0, 4.0])
    assert calculate_cindex(time, event, risk) == pytest.approx(0.0)


# ------------------------------------------------------------------- IBS

def test_calculate_ibs_works_on_current_numpy():
    """Regression: ``calculate_ibs`` crashed on NumPy >= 2.0.

    The implementation used ``getattr(np, "trapezoid", np.trapz)``.  The
    default argument is evaluated eagerly, so ``np.trapz`` raised
    ``AttributeError`` even though ``np.trapezoid`` exists -- the integration
    helper never resolved.  This test executes the integration path.
    """
    time, event = _toy_data()
    grid = evaluation_time_grid(time, points=40)
    value = calculate_ibs(grid, _constant_curve(grid, 0.5, len(time)), time, event)
    assert np.isfinite(value)
    assert value >= 0.0


def test_calculate_ibs_is_bounded_and_penalises_a_worse_fit():
    time, event = _toy_data()
    grid = evaluation_time_grid(time, points=60)
    n = len(time)
    # A curve close to the empirical survival law should score better than a
    # curve that decays far too fast.
    better = calculate_ibs(grid, _constant_curve(grid, 0.4, n), time, event)
    worse = calculate_ibs(grid, _constant_curve(grid, 4.0, n), time, event)
    assert 0.0 <= better <= 0.5
    assert 0.0 <= worse <= 1.0
    assert worse > better


def test_calculate_ibs_accepts_sample_major_orientation():
    time, event = _toy_data()
    grid = evaluation_time_grid(time, points=30)
    n = len(time)
    grid_major = _constant_curve(grid, 0.5, n)
    sample_major = grid_major.T
    assert calculate_ibs(grid, grid_major, time, event) == pytest.approx(
        calculate_ibs(grid, sample_major, time, event)
    )


@pytest.mark.parametrize(
    "grid,mutation",
    [
        (np.array([0.5]), "single point"),
        (np.array([2.0, 1.0]), "decreasing"),
    ],
)
def test_calculate_ibs_rejects_invalid_time_grid(grid, mutation):
    time, event = _toy_data(20)
    survival = np.ones((len(grid), len(time)))
    with pytest.raises(ValueError):
        calculate_ibs(grid, survival, time, event)


def test_calculate_ibs_rejects_non_binary_events():
    time, _ = _toy_data(20)
    grid = evaluation_time_grid(time, points=10)
    survival = np.ones((len(grid), len(time)))
    with pytest.raises(ValueError):
        calculate_ibs(grid, survival, time, np.full(len(time), 2.0))


# ------------------------------------------------- time-grid construction

def test_evaluation_time_grid_is_increasing_and_bounded():
    time, _ = _toy_data()
    grid = evaluation_time_grid(time, points=25)
    assert grid.shape == (25,)
    assert np.all(np.diff(grid) > 0)
    assert grid.min() >= np.quantile(time, 0.05)
    assert grid.max() <= np.quantile(time, 0.95)


def test_evaluation_time_grid_clips_to_survival_index():
    time, _ = _toy_data()
    index = np.linspace(0.5, 1.0, 20)
    grid = evaluation_time_grid(time, index, points=10)
    assert grid.min() >= index.min()
    assert grid.max() <= index.max()


def test_evaluation_time_grid_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluation_time_grid(np.array([]))
