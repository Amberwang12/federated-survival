"""Composable, inspectable client augmentation and comparison plots."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from ..core.augmenter import DataAugmenter
from .splitter import DataSet
from .workflow import _axis_values

ClientArrays = Dict[str, Tuple[np.ndarray, np.ndarray]]


def augment_clients(
    dataset: DataSet,
    method: str = "MVAEC",
    *,
    k: float = 0.5,
    latent_num: int = 10,
    hidden_num: int = 30,
    alpha: float = 1.0,
    beta: float = 1.0,
    sampling: str = "sparse",
    sparse_gamma: float = 0.1,
    seed: Optional[int] = 42,
) -> ClientArrays:
    """Return original-plus-synthetic client arrays without altering ``dataset``.

    MVAEC augments each client locally. MVAES pools generated samples before
    redistributing them; it must not be presented as a no-sharing workflow.
    The existing MVAE trainer uses 500 epochs per client. The output is for
    exploratory experiments, not a privacy guarantee. By default, synthetic
    samples are conditionally drawn near latent codes from the wider half of
    each client's uncensored-time distribution. Use
    ``sampling="unconditional"`` to recover prior-only latent sampling.
    """
    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be the result of partition_data(...)")
    selected = str(method).upper()
    if selected not in ("MVAEC", "MVAES"):
        raise ValueError("method must be MVAEC or MVAES")
    if not 0 < float(k) <= 1:
        raise ValueError("k must be in (0, 1]")
    if latent_num <= 0 or hidden_num <= 0 or alpha <= 0 or beta <= 0:
        raise ValueError("latent_num, hidden_num, alpha, and beta must be positive")
    if str(sampling).lower().replace("-", "_") not in {"sparse", "unconditional"}:
        raise ValueError("sampling must be 'sparse' or 'unconditional'")
    if sparse_gamma < 0:
        raise ValueError("sparse_gamma must be non-negative")

    augmenter = DataAugmenter(
        latent_num=int(latent_num),
        hidden_num=int(hidden_num),
        alpha=float(alpha),
        beta=float(beta),
        sampling=sampling,
        sparse_gamma=float(sparse_gamma),
    )
    if seed is None:
        return getattr(augmenter, selected.lower())(dataset.clients_set, k=float(k))

    # The legacy MVAE and redistribution code uses process-wide RNGs. Keep a
    # reproducible local call from changing the caller's random streams.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        return getattr(augmenter, selected.lower())(dataset.clients_set, k=float(k))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def summarize_augmentation(dataset: DataSet, augmented: ClientArrays):
    """Measure where decoded synthetic times landed relative to the target half.

    The returned DataFrame reports the sparse half selected from each client's
    original uncensored times and the fraction of appended synthetic rows whose
    decoded times actually fall in that half.
    """
    import pandas as pd

    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be the result of partition_data(...)")
    if set(augmented) != set(dataset.clients_set):
        raise ValueError("augmented clients must match the original client IDs")

    rows = []
    for client_id, (raw_x, raw_y) in dataset.clients_set.items():
        augmented_x, augmented_y = augmented[client_id]
        raw_x, raw_y = np.asarray(raw_x), np.asarray(raw_y)
        augmented_x, augmented_y = np.asarray(augmented_x), np.asarray(augmented_y)
        n_raw = len(raw_x)
        if (
            len(augmented_x) != len(augmented_y)
            or len(augmented_x) < n_raw
            or not np.array_equal(augmented_x[:n_raw], raw_x)
            or not np.array_equal(augmented_y[:n_raw], raw_y)
        ):
            raise ValueError("augmented arrays must start with unchanged original samples")

        event_times = raw_y[raw_y[:, 1].astype(int) == 1, 0]
        _, region = DataAugmenter._sparse_region(event_times)
        synthetic_times = augmented_y[n_raw:, 0].astype(float)
        if region["side"] == "late":
            in_target = synthetic_times > region["median"]
        else:
            in_target = synthetic_times <= region["median"]
        rows.append(
            {
                "client": client_id,
                "uncensored_median": region["median"],
                "target_region": region["side"],
                "source_count": region["source_count"],
                "synthetic_count": int(synthetic_times.size),
                "synthetic_in_target_count": int(in_target.sum()),
                "synthetic_in_target_fraction": (
                    float(in_target.mean()) if synthetic_times.size else np.nan
                ),
                "synthetic_time_min": (
                    float(np.min(synthetic_times)) if synthetic_times.size else np.nan
                ),
                "synthetic_time_median": (
                    float(np.median(synthetic_times)) if synthetic_times.size else np.nan
                ),
                "synthetic_time_max": (
                    float(np.max(synthetic_times)) if synthetic_times.size else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_augmentation_comparison(
    dataset: DataSet,
    augmentations: Mapping[str, ClientArrays],
    *,
    chart: str = "scatter",
    x: Optional[str] = None,
    y: str = "time",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """Compare generated samples from augmentation methods in separate panels.

    ``chart="strip"`` places clients on the x-axis and time on the y-axis,
    distinguishing original events, original censoring, and synthetic events.
    ``chart="scatter"`` uses feature/time axes and colours for clients.
    The held-out test set is never plotted or augmented.
    """
    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be the result of partition_data(...)")
    if not isinstance(augmentations, Mapping) or not augmentations:
        raise ValueError("augmentations must map method names to augmented client arrays")
    if chart not in ("scatter", "strip"):
        raise ValueError("chart must be scatter or strip")
    if chart == "strip" and (x is not None or y != "time"):
        raise ValueError("strip chart fixes x=client and y=time; omit x and y")
    scatter_x = x or "feature_1"
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(augmentations),
        figsize=(6.2 * len(augmentations), 5.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    try:
        for ax, (method, clients) in zip(axes[0], augmentations.items()):
            if set(clients) != set(dataset.clients_set):
                raise ValueError("augmented clients must match the original client IDs")
            synthetic_total = 0
            rng = np.random.RandomState(20260922)
            for index, (client_id, (raw_x, raw_y)) in enumerate(dataset.clients_set.items()):
                augmented_x, augmented_y = clients[client_id]
                raw_x, raw_y = np.asarray(raw_x), np.asarray(raw_y)
                augmented_x, augmented_y = np.asarray(augmented_x), np.asarray(augmented_y)
                n_raw = len(raw_x)
                if (
                    augmented_x.ndim != 2
                    or augmented_y.ndim != 2
                    or augmented_x.shape[1] != raw_x.shape[1]
                    or augmented_y.shape[1] != 2
                    or len(augmented_x) != len(augmented_y)
                    or len(augmented_x) < n_raw
                    or not np.array_equal(augmented_x[:n_raw], raw_x)
                    or not np.array_equal(augmented_y[:n_raw], raw_y)
                ):
                    raise ValueError("augmented arrays must start with unchanged original samples")
                synthetic_x, synthetic_y = augmented_x[n_raw:], augmented_y[n_raw:]
                synthetic_total += len(synthetic_x)
                if chart == "strip":
                    position = index + 1
                    for status, colour, marker in (
                        (1, "#0072B2", "o"),
                        (0, "#D55E00", "x"),
                    ):
                        selected = raw_y[:, 1].astype(int) == status
                        ax.scatter(
                            position + rng.uniform(-0.13, 0.13, int(selected.sum())),
                            raw_y[selected, 0],
                            color=colour,
                            marker=marker,
                            s=20,
                            alpha=0.58,
                            linewidths=0.8,
                        )
                    if len(synthetic_y):
                        ax.scatter(
                            position + rng.uniform(-0.13, 0.13, len(synthetic_y)),
                            synthetic_y[:, 0],
                            color="#009E73",
                            marker="^",
                            s=32,
                            alpha=0.85,
                        )
                else:
                    colour = palette(index)
                    ax.scatter(
                        _axis_values(raw_x, raw_y, scatter_x),
                        _axis_values(raw_x, raw_y, y),
                        s=13,
                        alpha=0.24,
                        color=colour,
                        marker="o",
                    )
                    if len(synthetic_x):
                        ax.scatter(
                            _axis_values(synthetic_x, synthetic_y, scatter_x),
                            _axis_values(synthetic_x, synthetic_y, y),
                            s=38,
                            alpha=0.85,
                            color=colour,
                            marker="x",
                            linewidths=1,
                        )
            ax.set_title("%s (%d synthetic)" % (method, synthetic_total), loc="left")
            if chart == "strip":
                ax.set_xticks(
                    range(1, len(dataset.clients_set) + 1),
                    ["Client %d" % (index + 1) for index in range(len(dataset.clients_set))],
                )
                ax.set_xlim(0.5, len(dataset.clients_set) + 0.5)
                ax.set_xlabel("Client")
            else:
                ax.set_xlabel("Time" if scatter_x == "time" else scatter_x.replace("_", " "))
            ax.grid(alpha=0.2)
        axes[0, 0].set_ylabel("Time" if y == "time" else str(y).replace("_", " "))
        if chart == "strip":
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    color="#0072B2",
                    label="Original, uncensored",
                ),
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="None",
                    color="#D55E00",
                    label="Original, censored",
                ),
                Line2D(
                    [],
                    [],
                    marker="^",
                    linestyle="None",
                    color="#009E73",
                    label="Synthetic, uncensored",
                ),
            ]
        else:
            handles = [
                Line2D(
                    [], [], marker="o", linestyle="None", color="gray", alpha=0.5, label="Original"
                ),
                Line2D([], [], marker="x", linestyle="None", color="gray", label="Synthetic"),
            ]
            handles.extend(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    color=palette(index),
                    label="Client %d" % (index + 1),
                )
                for index in range(len(dataset.clients_set))
            )
        figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return figure
    except Exception:
        plt.close(figure)
        raise


__all__ = ["augment_clients", "summarize_augmentation", "plot_augmentation_comparison"]
