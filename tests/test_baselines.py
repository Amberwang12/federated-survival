"""Descriptive summaries of the paired Center / FSA / Local comparison.

``summarize_results`` consumes the ``raw_results.csv`` layout written by
``experiments.workflow_runner``: one row per ``(split, model, method, seed)``
where ``method`` is ``Center``, ``FSA``, or ``Local`` -- plus per-client
``Local-client`` rows that must stay out of the aggregate.  It is a public
export of the ``experiments`` package and produces the paired deltas quoted in
the paper, so its arithmetic is pinned here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from federated_survival.experiments import summarize_results

SUMMARY_COLUMNS = {
    "split",
    "model",
    "method",
    "n",
    "c_index_mean",
    "c_index_sd",
    "ibs_mean",
    "ibs_sd",
    "fsa_minus_center_c_index",
    "fsa_minus_local_c_index",
    "fsa_minus_center_ibs",
    "fsa_minus_local_ibs",
}

DEFAULT_METHODS = (("Center", 0.62, 0.19), ("FSA", 0.60, 0.20), ("Local", 0.57, 0.22))


def _raw_rows(methods=DEFAULT_METHODS, seed=1, split="iid", model="DeepSurv"):
    return [
        {
            "split": split,
            "model": model,
            "seed": seed,
            "method": method,
            "client": "all",
            "n_train": 100,
            "c_index": c_index,
            "ibs": ibs,
        }
        for method, c_index, ibs in methods
    ]


def test_summarize_results_reports_paired_deltas():
    summary = summarize_results(pd.DataFrame(_raw_rows()))

    assert set(summary.columns) == SUMMARY_COLUMNS
    assert set(summary["method"]) == {"Center", "FSA", "Local"}
    assert (summary["n"] == 1).all()

    # The deltas are constant within a (split, model) group.
    deltas = summary[
        [
            "fsa_minus_center_c_index",
            "fsa_minus_local_c_index",
            "fsa_minus_center_ibs",
            "fsa_minus_local_ibs",
        ]
    ].drop_duplicates()
    assert len(deltas) == 1

    row = deltas.iloc[0]
    assert row["fsa_minus_center_c_index"] == pytest.approx(0.60 - 0.62)
    assert row["fsa_minus_local_c_index"] == pytest.approx(0.60 - 0.57)
    assert row["fsa_minus_center_ibs"] == pytest.approx(0.20 - 0.19)
    assert row["fsa_minus_local_ibs"] == pytest.approx(0.20 - 0.22)


def test_summarize_results_averages_across_seeds():
    second_seed = (
        ("Center", 0.60, 0.20),
        ("FSA", 0.60, 0.20),
        ("Local", 0.56, 0.24),
    )
    rows = _raw_rows(seed=1) + _raw_rows(methods=second_seed, seed=2)

    summary = summarize_results(pd.DataFrame(rows))

    assert (summary["n"] == 2).all()
    # seed 1: 0.60 - 0.62 = -0.02 ; seed 2: 0.60 - 0.60 = 0.0 -> mean -0.01
    assert summary["fsa_minus_center_c_index"].iloc[0] == pytest.approx(-0.01)
    # seed 1: 0.20 - 0.22 = -0.02 ; seed 2: 0.20 - 0.24 = -0.04 -> mean -0.03
    assert summary["fsa_minus_local_ibs"].iloc[0] == pytest.approx(-0.03)


def test_summarize_results_excludes_per_client_rows():
    rows = _raw_rows() + [
        {
            "split": "iid",
            "model": "DeepSurv",
            "seed": 1,
            "method": "Local-client",
            "client": "client0",
            "n_train": 50,
            "c_index": 0.51,
            "ibs": 0.30,
        }
    ]

    summary = summarize_results(pd.DataFrame(rows))

    assert set(summary["method"]) == {"Center", "FSA", "Local"}


def test_summarize_results_keeps_splits_and_models_separate():
    rows = _raw_rows(split="iid", model="DeepSurv")
    rows += _raw_rows(split="dirichlet", model="DeepHit", seed=3)

    summary = summarize_results(pd.DataFrame(rows))

    assert set(zip(summary["split"], summary["model"])) == {
        ("iid", "DeepSurv"),
        ("dirichlet", "DeepHit"),
    }
