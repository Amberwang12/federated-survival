import numpy as np
from typing import Optional
from lifelines.utils import concordance_index

def calculate_cindex(time: np.ndarray,
                    event: np.ndarray,
                    risk_score: np.ndarray) -> float:
    """
    Calculate the C-index

    Args:
        time: Survival times
        event: Event indicators
        risk_score: Risk scores

    Returns:
        float: C-index value
    """
    return concordance_index(time, -risk_score, event)

def calculate_ibs(time_grid: np.ndarray,
                 survival_curves: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray) -> float:
    """
    Calculate the Integrated Brier Score (IBS)

    Args:
        time_grid: Grid of time points
        survival_curves: Predicted survival curves
        time: Actual survival times
        event: Event indicators

    Returns:
        float: IBS value
    """
    grid = np.asarray(time_grid, dtype=float)
    durations = np.asarray(time, dtype=float).reshape(-1)
    events = np.asarray(event, dtype=float).reshape(-1)
    survival = np.asarray(survival_curves, dtype=float)
    if survival.shape == (len(durations), len(grid)):
        survival = survival.T
    if survival.shape != (len(grid), len(durations)):
        raise ValueError(
            "survival_curves must have shape (n_times, n_samples) or "
            "(n_samples, n_times)"
        )
    if len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("time_grid must contain at least two increasing values")
    if len(durations) == 0 or len(events) != len(durations):
        raise ValueError("time and event must be non-empty arrays of equal length")
    if not np.all(np.isin(events, [0.0, 1.0])):
        raise ValueError("event must contain only 0/1 indicators")

    # Kaplan--Meier estimator of G(t)=P(C>t), with censoring treated as event.
    order = np.argsort(durations, kind="mergesort")
    ordered_t = durations[order]
    ordered_censor = 1.0 - events[order]
    unique_t = np.unique(ordered_t)
    g_after = []
    g = 1.0
    for value in unique_t:
        at_risk = np.sum(ordered_t >= value)
        censored = ordered_censor[ordered_t == value].sum()
        if at_risk:
            g *= 1.0 - censored / at_risk
        g_after.append(g)
    g_after = np.asarray(g_after)

    def censor_survival(values, left_limit=False):
        side = "left" if left_limit else "right"
        positions = np.searchsorted(unique_t, values, side=side) - 1
        result = np.ones_like(np.asarray(values, dtype=float))
        valid = positions >= 0
        result[valid] = g_after[positions[valid]]
        return np.maximum(result, 1e-12)

    g_at_event_left = censor_survival(durations, left_limit=True)
    scores = np.empty(len(grid), dtype=float)
    for index, value in enumerate(grid):
        failed = (durations <= value) & (events == 1)
        still_at_risk = durations > value
        weights_failed = failed / g_at_event_left
        weights_risk = still_at_risk / censor_survival(np.full(len(durations), value))
        scores[index] = np.mean(
            weights_failed * survival[index] ** 2
            + weights_risk * (1.0 - survival[index]) ** 2
        )
    # NumPy 2.0 renamed ``trapz`` to ``trapezoid`` and removed the old name,
    # so the fallback must be resolved lazily: passing ``np.trapz`` as the
    # ``getattr`` default would raise AttributeError on NumPy >= 2.0.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(scores, grid) / (grid[-1] - grid[0]))


def evaluation_time_grid(
    durations: np.ndarray,
    survival_index: Optional[np.ndarray] = None,
    quantiles=(0.05, 0.95),
    points: int = 100,
) -> np.ndarray:
    """Return a stable evaluation interval away from unsupported censoring tails."""
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        raise ValueError("durations cannot be empty")
    lower, upper = np.quantile(durations, quantiles)
    if survival_index is not None:
        index = np.asarray(survival_index, dtype=float)
        lower = max(float(lower), float(index.min()))
        upper = min(float(upper), float(index.max()))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower, upper = float(durations.min()), float(durations.max())
    if lower >= upper:
        raise ValueError("evaluation time range is degenerate")
    return np.linspace(lower, upper, points)
