"""Tests for the real datasets bundled inside the installed package."""

import pandas as pd
import pytest

import federated_survival as fs


def test_available_real_datasets_includes_bundled_tables():
    names = fs.available_real_datasets()
    assert "gbsg" in names
    assert "colon_death" in names


def test_load_real_data_gbsg_returns_canonical_table():
    data = fs.load_real_data("gbsg")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (686, 10)
    assert list(data.columns[-2:]) == ["time", "status"]
    assert data["status"].isin([0, 1]).all()
    assert data["time"].dtype == "float64"


def test_load_real_data_colon_death():
    data = fs.load_real_data("colon_death")
    assert data.shape == (888, 13)
    assert list(data.columns[-2:]) == ["time", "status"]


def test_load_real_data_unknown_name_lists_available():
    with pytest.raises(FileNotFoundError, match="Available:"):
        fs.load_real_data("does_not_exist")


def test_load_real_data_rejects_path_segments():
    with pytest.raises(ValueError, match="Invalid dataset name"):
        fs.load_real_data("../data/real/gbsg")
