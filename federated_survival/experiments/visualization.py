"""Deterministic, dependency-light visualizations for saved experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "Center": "#4C78A8",
    "Federated": "#009E73",
    "Weighted Local": "#E45756",
    "event": "#0072B2",
    "censored": "#D55E00",
}
METHOD_LABELS = {
    "Center": "Center",
    "FSA": "Federated",
    "Local": "Weighted Local",
}


def _apply_style(style: str) -> None:
    size = 11 if style == "presentation" else 9
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": size,
            "axes.titlesize": size + 1,
            "axes.labelsize": size,
            "xtick.labelsize": size - 1,
            "ytick.labelsize": size - 1,
            "legend.fontsize": size - 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, directory: Path, stem: str, formats: Iterable[str]) -> List[Path]:
    paths = []
    directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = directory / (stem + "." + extension)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _method_frame(raw: pd.DataFrame, requested: Sequence[str]) -> pd.DataFrame:
    frame = raw[raw["method"].isin(METHOD_LABELS)].copy()
    frame["display_method"] = frame["method"].map(METHOD_LABELS)
    unknown = sorted(set(requested) - set(METHOD_LABELS.values()))
    if unknown:
        raise ValueError("unknown visualization methods: %s" % unknown)
    frame = frame[frame["display_method"].isin(requested)]
    return frame


def _draw_partition(ax, observations: pd.DataFrame, settings: Dict[str, object]) -> None:
    first_seed = int(observations["seed"].min())
    frame = observations[observations["seed"].eq(first_seed)].copy()
    clients = list(dict.fromkeys(frame["client"].tolist()))
    if all(str(client).startswith("client") and str(client)[6:].isdigit() for client in clients):
        client_labels = ["Client %d" % (int(str(client)[6:]) + 1) for client in clients]
    else:
        client_labels = [str(client) for client in clients]
    plot_type = str(settings["type"])
    values = [frame.loc[frame["client"].eq(client), "time"].to_numpy() for client in clients]

    if plot_type == "scatter":
        x_column = str(settings["x"])
        y_column = str(settings["y"])
        missing = [column for column in (x_column, y_column) if column not in frame.columns]
        if missing:
            raise ValueError(
                "scatter axes are unavailable for this dataset: %s; available features are %s"
                % (missing, [column for column in frame if column.startswith("feature_")])
            )
        palette = plt.get_cmap("tab10")
        for client_index, client in enumerate(clients):
            client_frame = frame[frame["client"].eq(client)]
            display_client = client_labels[client_index]
            details = []
            if settings.get("show_sample_size"):
                details.append("n=%d" % len(client_frame))
            if settings.get("show_censoring_rate"):
                details.append("censored=%.1f%%" % (100 * (1 - client_frame["status"].mean())))
            if details:
                display_client += " (%s)" % "; ".join(details)
            for status, status_label, marker in ((1, "Event", "o"), (0, "Censored", "x")):
                selected = client_frame[client_frame["status"].eq(status)]
                ax.scatter(
                    selected[x_column],
                    selected[y_column],
                    s=24,
                    marker=marker,
                    alpha=0.68,
                    linewidths=0.7,
                    color=palette(client_index),
                    label="%s - %s" % (display_client, status_label),
                )
        ax.set_xlabel("Observed time" if x_column == "time" else x_column.replace("_", " "))
        ax.set_ylabel("Observed time" if y_column == "time" else y_column.replace("_", " "))
        ax.legend(loc="best", frameon=True, ncol=2)
    elif plot_type == "strip":
        rng = np.random.RandomState(20260921)
        for index, client in enumerate(clients, start=1):
            client_frame = frame[frame["client"].eq(client)]
            for status, label, marker in ((1, "Uncensored", "o"), (0, "Censored", "x")):
                selected = client_frame[client_frame["status"].eq(status)]
                jitter = rng.uniform(-0.13, 0.13, len(selected))
                ax.scatter(
                    np.full(len(selected), index) + jitter,
                    selected["time"],
                    s=14,
                    marker=marker,
                    alpha=0.68,
                    linewidths=0.7,
                    color=COLORS["event" if status == 1 else "censored"],
                    label=label if index == 1 else None,
                )
        ax.legend(loc="lower right", frameon=True)
    elif plot_type == "boxplot":
        positions = np.arange(1, len(clients) + 1)
        boxes = ax.boxplot(values, positions=positions, patch_artist=True, showmeans=True)
        ax.set_xticks(positions, client_labels)
        for box in boxes["boxes"]:
            box.set_facecolor("#BBD7EA")
    elif plot_type == "violin":
        violins = ax.violinplot(values, positions=np.arange(1, len(clients) + 1), showmeans=True)
        for body in violins["bodies"]:
            body.set_facecolor("#4C78A8")
            body.set_alpha(0.45)
        ax.set_xticks(np.arange(1, len(clients) + 1), client_labels)
    else:
        rates = [
            1.0 - float(frame.loc[frame["client"].eq(client), "status"].mean())
            for client in clients
        ]
        ax.bar(client_labels, rates, color="#D55E00", alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Censoring rate")
        for index, value in enumerate(rates):
            ax.text(index, value + 0.02, "%.1f%%" % (100 * value), ha="center")

    if plot_type not in ("censoring_bar", "scatter"):
        ax.set_xticks(np.arange(1, len(clients) + 1), client_labels)
        ax.set_ylabel("Observed time")
    if plot_type != "scatter" and (
        settings.get("show_sample_size") or settings.get("show_censoring_rate")
    ):
        labels = []
        for client in clients:
            selected = frame[frame["client"].eq(client)]
            parts = []
            if settings.get("show_sample_size"):
                parts.append("n=%d" % len(selected))
            if settings.get("show_censoring_rate"):
                parts.append("censored=%.1f%%" % (100 * (1 - selected["status"].mean())))
            labels.append("; ".join(parts))
        if plot_type != "censoring_bar":
            ymax = float(frame["time"].max())
            for index, label in enumerate(labels, start=1):
                ax.text(index, ymax * 1.03, label, ha="center", va="bottom", fontsize=8)
            ax.set_ylim(top=ymax * 1.16)
    ax.set_title("Client partition (seed %d)" % first_seed, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def _interval(values: np.ndarray, mode: str) -> float:
    if len(values) <= 1 or mode == "none":
        return 0.0
    standard_deviation = float(np.std(values, ddof=1))
    if mode == "mean_sd":
        return standard_deviation
    return 1.96 * standard_deviation / np.sqrt(len(values))


def _draw_round_metric(ax, rounds: pd.DataFrame, metric: str, aggregate: str) -> None:
    column = "test_c_index" if metric == "c_index" else "test_ibs"
    color = "#0072B2" if metric == "c_index" else "#D55E00"
    if aggregate == "none":
        for seed, seed_frame in rounds.groupby("seed", sort=True):
            seed_frame = seed_frame.sort_values("round")
            ax.plot(
                seed_frame["round"],
                seed_frame[column],
                color=color,
                alpha=0.35,
                linewidth=1.1,
                label="seed %s" % seed,
            )
        means = None
    else:
        means = True
    grouped = rounds.groupby("round", sort=True)[column]
    x = np.asarray(sorted(rounds["round"].unique()), dtype=float)
    mean_values = grouped.mean().reindex(x.astype(int)).to_numpy(dtype=float)
    half = np.asarray(
        [
            _interval(grouped.get_group(int(round_number)).to_numpy(dtype=float), aggregate)
            for round_number in x
        ]
    )
    if means is not None:
        ax.plot(x, mean_values, color=color, marker="o", linewidth=1.8)
        if np.any(half > 0):
            ax.fill_between(x, mean_values - half, mean_values + half, color=color, alpha=0.18)
    label = "Test C-index" if metric == "c_index" else "Test IBS"
    ax.set_title(label, loc="left", fontweight="bold")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("C-index (higher is better)" if metric == "c_index" else "IBS (lower is better)")
    ax.grid(alpha=0.25)


def _draw_final_metric(
    ax,
    raw: pd.DataFrame,
    metric: str,
    settings: Dict[str, object],
    compact: bool = False,
) -> None:
    requested = list(settings["methods"])
    frame = _method_frame(raw, requested)
    labels = [label for label in requested if label in set(frame["display_method"])]
    values = [
        frame.loc[frame["display_method"].eq(label), metric].to_numpy(dtype=float)
        for label in labels
    ]
    plot_type = str(settings["type"])
    positions = np.arange(1, len(labels) + 1)
    tick_labels = (
        [
            {"Center": "Center", "Federated": "Fed.", "Weighted Local": "Local"}[label]
            for label in labels
        ]
        if compact
        else labels
    )

    if plot_type == "boxplot":
        boxes = ax.boxplot(values, positions=positions, patch_artist=True, showmeans=False)
        ax.set_xticks(positions, tick_labels)
        for box, label in zip(boxes["boxes"], labels):
            box.set_facecolor(COLORS[label])
            box.set_alpha(0.55)
    elif plot_type == "violin":
        violins = ax.violinplot(values, positions=positions, showmeans=False, showextrema=True)
        for body, label in zip(violins["bodies"], labels):
            body.set_facecolor(COLORS[label])
            body.set_alpha(0.5)
        ax.set_xticks(positions, tick_labels)
    elif plot_type in ("dotplot", "bar"):
        means = np.asarray([np.mean(value) for value in values])
        errors = np.asarray([_interval(value, "mean_ci95") for value in values])
        if plot_type == "bar":
            ax.bar(
                positions,
                means,
                yerr=errors,
                color=[COLORS[label] for label in labels],
                alpha=0.7,
                capsize=4,
            )
        else:
            ax.errorbar(positions, means, yerr=errors, fmt="o", color="#263238", capsize=4)
        ax.set_xticks(positions, tick_labels)
    else:
        summaries = [
            [
                label,
                "%.4f" % np.mean(value),
                "%.4f" % np.std(value, ddof=1) if len(value) > 1 else "--",
            ]
            for label, value in zip(labels, values)
        ]
        ax.axis("off")
        table = ax.table(
            cellText=summaries,
            colLabels=["Method", "Mean", "SD"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

    if plot_type != "table" and settings.get("show_raw_points"):
        rng = np.random.RandomState(20260921)
        for position, value, label in zip(positions, values, labels):
            jitter = rng.uniform(-0.07, 0.07, len(value))
            ax.scatter(
                position + jitter,
                value,
                color=COLORS[label],
                edgecolor="white",
                linewidth=0.35,
                s=25,
                zorder=4,
            )
    if plot_type != "table" and settings.get("show_mean"):
        for position, value in zip(positions, values):
            ax.scatter(position, np.mean(value), marker="D", color="black", s=28, zorder=5)
    if plot_type != "table":
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
        ax.set_ylabel(
            "C-index (higher is better)" if metric == "c_index" else "IBS (lower is better)"
        )
    ax.set_title(
        "C-index" if metric == "c_index" else "Integrated Brier score",
        loc="left",
        fontweight="bold",
    )


def plot_client_partition(
    observations: pd.DataFrame,
    settings: Dict[str, object],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _draw_partition(ax, observations, settings)
    fig.tight_layout()
    return _save(fig, output_dir, "client_partition", formats)


def plot_round_metrics(
    rounds: pd.DataFrame,
    settings: Dict[str, object],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric in zip(axes, ("c_index", "ibs")):
        _draw_round_metric(ax, rounds, metric, str(settings["aggregate_seeds"]))
    fig.tight_layout()
    return _save(fig, output_dir, "round_metrics", formats)


def plot_final_metrics(
    raw: pd.DataFrame,
    settings: Dict[str, object],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    metrics = list(settings["metrics"])
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _draw_final_metric(ax, raw, metric, settings)
    fig.tight_layout()
    return _save(fig, output_dir, "final_metrics_%s" % settings["type"], formats)


def plot_composite(
    raw: pd.DataFrame,
    rounds: pd.DataFrame,
    observations: pd.DataFrame,
    configuration: Dict[str, object],
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    visualization = configuration["visualization"]
    fig = plt.figure(figsize=(12.4, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])

    model = configuration["model"]
    federated = configuration["federated"]
    partition = configuration["partition"]
    data = configuration["data"]
    dataset_label = str(data.get("label", configuration["name"]))

    _draw_partition(ax_a, observations, visualization["client_partition"])
    ax_a.set_title(
        "(A) %s data and %s client partition" % (dataset_label, str(partition["method"]).upper()),
        loc="left",
        fontweight="bold",
    )

    ax_b.axis("off")
    lines = [
        "dataset:          %s" % dataset_label,
        "model:            %s" % model["name"],
        "protocol:         %s" % federated["protocol"],
        "clients:          %s (%s; %s participation)"
        % (
            partition["n_clients"],
            str(partition["method"]).upper(),
            "full" if federated["client_fraction"] == 1.0 else "partial",
        ),
        "global_rounds:    %s" % federated["global_rounds"],
        "local_steps:      %s optimizer updates / round" % federated["local_steps"],
        "batch_size:       %s" % federated["batch_size"],
        "optimizer:        %s (learning_rate = %s)"
        % (str(federated["optimizer"]).title(), federated["learning_rate"]),
    ]
    lines.append("seeds:            %s" % len(configuration["experiment"]["seeds"]))
    lines.append(
        "reference_budget: %s optimizer updates" % configuration["references"]["optimizer_steps"]
    )
    ax_b.text(
        0.04,
        0.90,
        "\n".join(lines),
        va="top",
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F3F6F9", edgecolor="#AEB8C4"),
    )
    ax_b.set_title("(B) Reproducible configuration", loc="left", fontweight="bold")

    final_settings = dict(visualization["final_metrics"])
    right = grid[1, 1].subgridspec(1, 2, wspace=0.42)
    ax_d1 = fig.add_subplot(right[0, 0])
    ax_d2 = fig.add_subplot(right[0, 1])
    _draw_round_metric(
        ax_c,
        rounds,
        "c_index",
        str(visualization["round_metrics"]["aggregate_seeds"]),
    )
    ax_c.set_title("(C) Round-wise test C-index", loc="left", fontweight="bold")
    _draw_final_metric(ax_d1, raw, "c_index", final_settings, compact=True)
    _draw_final_metric(ax_d2, raw, "ibs", final_settings, compact=True)
    ax_d1.set_title("(D1) C-index", loc="left", fontweight="bold")
    ax_d2.set_title("(D2) IBS", loc="left", fontweight="bold")
    fig.suptitle(configuration["name"], fontweight="bold")
    return _save(fig, output_dir, "overview", formats)


def generate_visualizations(
    raw: pd.DataFrame,
    rounds: pd.DataFrame,
    observations: pd.DataFrame,
    configuration: Dict[str, object],
    output_dir: Path,
) -> List[Path]:
    """Generate every visualization requested by one normalized configuration."""
    settings = configuration["visualization"]
    if not settings["enabled"]:
        return []
    _apply_style(str(settings["style"]))
    formats = list(settings["formats"])
    figure_dir = output_dir / "figures"
    paths = []
    if not observations.empty:
        paths.extend(
            plot_client_partition(observations, settings["client_partition"], figure_dir, formats)
        )
    if not rounds.empty:
        paths.extend(plot_round_metrics(rounds, settings["round_metrics"], figure_dir, formats))
    if not raw.empty:
        paths.extend(plot_final_metrics(raw, settings["final_metrics"], figure_dir, formats))
    if settings["composite"]["enabled"]:
        if rounds.empty or observations.empty:
            raise ValueError("composite visualization requires round metrics and partition data")
        paths.extend(plot_composite(raw, rounds, observations, configuration, figure_dir, formats))
    return paths
