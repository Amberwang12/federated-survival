from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import federated_survival as fs


def test_top_level_data_workflow_generates_and_partitions():
    first = fs.simulate_data(n_samples=90, n_features=3, seed=12)
    second = fs.simulate_data(n_samples=90, n_features=3, seed=12)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["x1", "x2", "x3", "time", "status"]

    dataset = fs.partition_data(first, n_clients=3, method="iid", seed=12)
    assert len(dataset.train_data) == 72
    assert len(dataset.test_data) == 18
    assert sum(len(features) for features, _ in dataset.clients_set.values()) == 72


def test_top_level_partition_plot_saves_figure(tmp_path):
    data = fs.simulate_data(n_samples=90, n_features=3, seed=13)
    dataset = fs.partition_data(data, n_clients=3, method="iid", seed=13)
    output = tmp_path / "partition.png"

    figure = fs.plot_partition(
        dataset,
        x="feature_1",
        y="time",
        output_path=output,
        show=False,
    )

    assert output.exists()
    assert len(figure.axes) == 1
    plt.close(figure)


def test_client_strip_plot_groups_event_and_censoring_by_client(tmp_path):
    data = fs.simulate_data(n_samples=90, n_features=3, seed=13)
    dataset = fs.partition_data(data, n_clients=3, method="iid", seed=13)
    output = tmp_path / "clients.png"

    figure = fs.plot_partition(dataset, chart="strip", output_path=output, show=False)

    assert output.exists()
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Client"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Client 1",
        "Client 2",
        "Client 3",
    ]
    assert [item.get_text() for item in axis.get_legend().get_texts()] == [
        "Uncensored",
        "Censored",
    ]
    assert len(axis.collections) == 6
    assert len(axis.texts) == 3
    plt.close(figure)

    with pytest.raises(ValueError, match="partition chart"):
        fs.plot_partition(dataset, chart="unknown", show=False)


def test_top_level_load_data_uses_canonical_names(tmp_path):
    source = tmp_path / "survival.csv"
    pd.DataFrame(
        {
            "age": [50, 60],
            "marker": [0.1, 0.2],
            "follow_up": [3.0, 5.0],
            "event": [1, 0],
        }
    ).to_csv(source, index=False)

    loaded = fs.load_data(source, duration_column="follow_up", event_column="event")

    assert list(loaded.columns) == ["x1", "x2", "time", "status"]


def test_partition_data_many_returns_seeded_partitions():
    data = fs.simulate_data(n_samples=90, n_features=3, seed=12)
    partitions = fs.partition_data_many(data, seeds=[0, 1], n_clients=3)
    assert list(partitions) == [0, 1]
    assert all(len(split.train_data) == 72 for split in partitions.values())


def test_real_data_preprocessing_uses_training_statistics_only(tmp_path):
    source = tmp_path / "survival.csv"
    frame = pd.DataFrame(
        {
            "marker": [float(index) for index in range(35)] + [np.nan],
            "follow_up": [float(index + 1) for index in range(36)],
            "event": [index % 2 for index in range(36)],
        }
    )
    frame.to_csv(source, index=False)
    data = fs.load_data(source, duration_column="follow_up", event_column="event")
    with pytest.raises(ValueError, match="missing values"):
        fs.partition_data(data, n_clients=3, seed=0)
    raw = fs.partition_data(data, n_clients=3, seed=0, missing_values="median")
    processed = fs.partition_data(
        data, n_clients=3, seed=0, missing_values="median", standardize=True
    )
    train_median = np.nanmedian(raw.train_data[:, 0])
    imputed_train = np.where(np.isnan(raw.train_data[:, 0]), train_median, raw.train_data[:, 0])
    train_mean = np.mean(imputed_train)
    train_sd = np.std(imputed_train)
    expected_test = np.where(np.isnan(raw.test_data[:, 0]), train_median, raw.test_data[:, 0])
    np.testing.assert_allclose(
        processed.test_data[:, 0], (expected_test - train_mean) / train_sd, atol=1e-6
    )
    np.testing.assert_allclose(processed.train_data.mean(axis=0), 0, atol=1e-6)
    assert np.isfinite(processed.train_data).all()
    assert np.isfinite(processed.test_data).all()


def test_partition_data_many_passes_preprocessing_options():
    data = fs.simulate_data(n_samples=90, n_features=3, seed=12)
    splits = fs.partition_data_many(data, seeds=[0, 1], standardize=True)
    for split in splits.values():
        np.testing.assert_allclose(split.train_data.mean(axis=0), 0, atol=1e-6)
