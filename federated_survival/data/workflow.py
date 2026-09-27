"""Convenience functions for interactive data preparation and inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .generator import DataGenerator, SimulationConfig
from .loader import DataLoader
from .preprocessing import preprocess_partition
from .splitter import DataSet, DataSplitter


def simulate_data(
    mechanism: str = "weibull",
    n_samples: int = 300,
    n_features: int = 5,
    censoring: float = 0.4,
    seed: Optional[int] = 42,
    *,
    u_max: float = 4.0,
    c_step: float = 0.4,
) -> pd.DataFrame:
    """Generate one simulated survival table through the public package API.

    ``censoring`` is the censoring-distribution control used by Weibull,
    log-normal, and SDGM1 simulations; it is not an exact requested censoring
    proportion. SDGM2/SDGM3 use ``u_max`` and SDGM4 uses ``c_step``.
    """
    if int(n_samples) <= 0 or int(n_features) <= 0:
        raise ValueError("n_samples and n_features must be positive")
    generator = DataGenerator(
        SimulationConfig(
            n_samples=int(n_samples),
            n_features=int(n_features),
            random_state=seed,
        )
    )
    return generator.generate(
        mechanism,
        c_mean=float(censoring),
        u_max=float(u_max),
        c_step=float(c_step),
    )


def load_data(
    path: Union[str, Path],
    duration_column: str = "time",
    event_column: str = "status",
) -> pd.DataFrame:
    """Load CSV/Excel survival data and return canonical x1..xp/time/status columns."""
    if duration_column == event_column:
        raise ValueError("duration_column and event_column must be different")
    frame = DataLoader(
        time_column=duration_column,
        status_column=event_column,
    ).load(path)
    return frame.rename(columns={duration_column: "time", event_column: "status"})


def partition_data(
    data: pd.DataFrame,
    n_clients: int = 3,
    method: str = "iid",
    test_size: float = 0.2,
    seed: Optional[int] = 42,
    alpha: float = 0.5,
    *,
    standardize: bool = False,
    missing_values: str = "error",
) -> DataSet:
    """Split data and optionally preprocess features from training statistics only."""
    if missing_values not in ("error", "drop", "median"):
        raise ValueError("missing_values must be error, drop, or median")
    feature_columns = [column for column in data if column not in ("time", "status")]
    if missing_values == "error" and data[feature_columns].isna().any().any():
        raise ValueError("feature columns contain missing values; choose drop or median")
    if missing_values == "drop":
        data = data.dropna(axis=0).reset_index(drop=True)
    dataset = DataSplitter(
        n_clients=int(n_clients),
        split_type=method,
        alpha=float(alpha),
        test_size=float(test_size),
        random_state=seed,
    ).split(data)
    return preprocess_partition(dataset, missing_values=missing_values, standardize=standardize)


def partition_data_many(
    data: pd.DataFrame,
    seeds: Sequence[int],
    n_clients: int = 3,
    method: str = "iid",
    test_size: float = 0.2,
    alpha: float = 0.5,
    *,
    standardize: bool = False,
    missing_values: str = "error",
) -> Dict[int, DataSet]:
    """Create paired client partitions for multiple random seeds."""
    if isinstance(seeds, (str, bytes)) or not seeds:
        raise ValueError("seeds must be a non-empty sequence of integers")
    parsed = [int(seed) for seed in seeds]
    if len(set(parsed)) != len(parsed):
        raise ValueError("seeds must not contain duplicates")
    return {
        seed: partition_data(
            data,
            n_clients=n_clients,
            method=method,
            test_size=test_size,
            seed=seed,
            alpha=alpha,
            standardize=standardize,
            missing_values=missing_values,
        )
        for seed in parsed
    }


def _axis_values(features: np.ndarray, labels: np.ndarray, axis: str) -> np.ndarray:
    if axis == "time":
        return labels[:, 0]
    rendered = str(axis).lower()
    if rendered.startswith("feature_"):
        rendered = rendered[8:]
    elif rendered.startswith("x"):
        rendered = rendered[1:]
    if not rendered.isdigit() or int(rendered) <= 0:
        raise ValueError("scatter axes must be time, feature_N, or xN")
    index = int(rendered) - 1
    if index >= features.shape[1]:
        raise ValueError(
            "requested feature %d but the partition has %d features"
            % (index + 1, features.shape[1])
        )
    return features[:, index]


def _draw_partition(ax, dataset: DataSet, x: str, y: str, chart: str) -> None:
    """Draw one partition on an existing Matplotlib axis."""
    if chart not in ("scatter", "strip"):
        raise ValueError("partition chart must be scatter or strip")
    import matplotlib.pyplot as plt

    if chart == "strip":
        rng = np.random.RandomState(20260921)
        client_labels = []
        ymax = max(
            float(np.asarray(labels)[:, 0].max()) for _, labels in dataset.clients_set.values()
        )
        for client_index, (_, labels) in enumerate(dataset.clients_set.values(), start=1):
            labels = np.asarray(labels)
            client_labels.append("Client %d" % client_index)
            for status, name, marker, color in (
                (1, "Uncensored", "o", "#0072B2"),
                (0, "Censored", "x", "#D55E00"),
            ):
                selected = labels[:, 1].astype(int) == status
                positions = client_index + rng.uniform(-0.12, 0.12, int(selected.sum()))
                ax.scatter(
                    positions,
                    labels[selected, 0],
                    color=color,
                    marker=marker,
                    alpha=0.7,
                    s=25,
                    linewidths=0.7,
                    label=name if client_index == 1 else None,
                )
            ax.text(
                client_index,
                ymax * 1.05,
                "n = %d\ncensored = %.1f%%" % (len(labels), 100 * (1 - float(labels[:, 1].mean()))),
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="#C9D6E2", alpha=0.9),
            )
        ax.set_xticks(range(1, len(client_labels) + 1), client_labels)
        ax.set_xlim(0.5, len(client_labels) + 0.5)
        ax.set_ylim(top=ymax * 1.22)
        ax.set_xlabel("Client")
        ax.set_ylabel("Observed time")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=True)
        ax.grid(axis="y", alpha=0.25)
    else:
        palette = plt.get_cmap("tab10")
        for client_index, (client, (features, labels)) in enumerate(dataset.clients_set.items()):
            features = np.asarray(features)
            labels = np.asarray(labels)
            x_values = _axis_values(features, labels, x)
            y_values = _axis_values(features, labels, y)
            censoring_rate = 100 * (1 - labels[:, 1].mean())
            display_client = "Client %d (n=%d; censored=%.1f%%)" % (
                client_index + 1,
                len(labels),
                censoring_rate,
            )
            for status, status_label, marker in ((1, "Event", "o"), (0, "Censored", "x")):
                selected = labels[:, 1].astype(int) == status
                ax.scatter(
                    x_values[selected],
                    y_values[selected],
                    color=palette(client_index),
                    marker=marker,
                    alpha=0.68,
                    s=28,
                    linewidths=0.7,
                    label="%s - %s" % (display_client, status_label),
                )
        ax.set_xlabel("Observed time" if x == "time" else str(x).replace("_", " "))
        ax.set_ylabel("Observed time" if y == "time" else str(y).replace("_", " "))
        ax.legend(loc="best", frameon=True, ncol=2, fontsize=8)
        ax.grid(alpha=0.25)
    ax.set_title("Federated client partition", loc="left", fontweight="bold")


def plot_partition(
    dataset: DataSet,
    x: str = "feature_1",
    y: str = "time",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    *,
    chart: str = "scatter",
):
    """Draw a feature scatter or a client-wise event/censoring strip plot.

    The returned object is a Matplotlib ``Figure``. Set ``show=False`` for
    scripts, tests, or headless execution and use ``output_path`` to save it.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _draw_partition(ax, dataset, x, y, chart)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = [
    "simulate_data",
    "load_data",
    "partition_data",
    "partition_data_many",
    "plot_partition",
]
