"""Public, composable data-augmentation calls and comparison plots."""

import random

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import federated_survival as fs  # noqa: E402
from federated_survival.core.augmenter import DataAugmenter  # noqa: E402


@pytest.fixture
def partition():
    frame = fs.simulate_data(n_samples=120, n_features=2, seed=10)
    return fs.partition_data(frame, n_clients=3, seed=10)


@pytest.fixture
def fast_augmentation(monkeypatch):
    def fake_aug_client(self, features, labels, k):
        n = int(np.sum(labels[:, 1] == 1) * k)
        synthetic_features = np.full((n, features.shape[1]), 2.0, dtype=np.float32)
        synthetic_labels = np.column_stack(
            (np.full(n, float(np.mean(labels[:, 0]))), np.ones(n))
        ).astype(np.float32)
        return synthetic_features, synthetic_labels

    monkeypatch.setattr(DataAugmenter, "_aug_client", fake_aug_client)


def test_augment_clients_reproducible_and_preserves_original(partition, fast_augmentation):
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    local = fs.augment_clients(partition, method="MVAEC", k=0.5, seed=5)
    pooled = fs.augment_clients(partition, method="MVAES", k=0.5, seed=5)
    repeated = fs.augment_clients(partition, method="MVAES", k=0.5, seed=5)

    for client_id, (original_x, original_y) in partition.clients_set.items():
        for augmented in (local, pooled):
            features, labels = augmented[client_id]
            assert np.array_equal(features[: len(original_x)], original_x)
            assert np.array_equal(labels[: len(original_y)], original_y)
            assert len(features) > len(original_x)
        assert np.array_equal(pooled[client_id][0], repeated[client_id][0])
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.random.get_rng_state(), before_torch)


def test_plot_two_augmentation_methods(partition, fast_augmentation, tmp_path):
    methods = {
        method: fs.augment_clients(partition, method=method, k=0.5, seed=5)
        for method in ("MVAEC", "MVAES")
    }
    target = tmp_path / "augmented_scatter.png"
    figure = fs.plot_augmentation_comparison(
        partition, methods, x="feature_1", y="time", output_path=target, show=False
    )
    assert target.exists()
    assert len(figure.axes) == 2
    assert "MVAEC" in figure.axes[0].get_title(loc="left")
    assert "MVAES" in figure.axes[1].get_title(loc="left")
    plt.close(figure)

    client_figure = fs.plot_augmentation_comparison(
        partition,
        methods,
        chart="strip",
        output_path=tmp_path / "augmented_clients.png",
        show=False,
    )
    assert (tmp_path / "augmented_clients.png").exists()
    for axis in client_figure.axes:
        assert axis.get_xlabel() == "Client"
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "Client 1",
            "Client 2",
            "Client 3",
        ]
    plt.close(client_figure)

    summary = fs.summarize_augmentation(partition, methods["MVAEC"])
    assert list(summary["client"]) == list(partition.clients_set)
    assert (summary["synthetic_count"] > 0).all()
    assert summary["synthetic_in_target_fraction"].between(0, 1).all()

    with pytest.raises(ValueError, match="fixes x=client"):
        fs.plot_augmentation_comparison(
            partition, methods, chart="strip", x="feature_2", show=False
        )


def test_augmentation_calls_reject_invalid_settings(partition, fast_augmentation):
    with pytest.raises(ValueError, match="method must be"):
        fs.augment_clients(partition, method="unknown")
    with pytest.raises(ValueError, match="k must be"):
        fs.augment_clients(partition, k=0)
    with pytest.raises(ValueError, match="sampling must be"):
        fs.augment_clients(partition, sampling="latest")
    with pytest.raises(ValueError, match="sparse_gamma"):
        fs.augment_clients(partition, sparse_gamma=-0.1)
    clients = fs.augment_clients(partition, method="MVAEC", seed=4)
    client_id = next(iter(clients))
    clients[client_id] = (clients[client_id][0][1:], clients[client_id][1][1:])
    with pytest.raises(ValueError, match="unchanged original"):
        fs.plot_augmentation_comparison(partition, {"MVAEC": clients}, show=False)


def test_mvaes_accepts_nondefault_client_ids(partition, fast_augmentation):
    renamed = partition._replace(
        clients_set={
            "hospital_%d" % index: arrays
            for index, arrays in enumerate(partition.clients_set.values())
        }
    )
    augmented = fs.augment_clients(renamed, method="MVAES", k=0.5, seed=5)
    assert set(augmented) == set(renamed.clients_set)
