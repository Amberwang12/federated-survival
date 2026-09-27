"""Composable public API for paired survival-method experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch.nn as nn

from .core.config import FSAConfig
from .data.splitter import DataSet
from .data.workflow import _draw_partition, plot_partition
from .experiments.baselines import MODEL_NAMES, run_paired_baselines

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
}
PROTOCOL_PARAMETER_ALIASES = {
    "mu": "proximal_mu",
    "optimizer": "server_optimizer",
    "server_lr": "server_learning_rate",
    "beta1": "server_beta1",
    "beta2": "server_beta2",
    "tau": "server_tau",
    "max_iter": "webdisco_max_iter",
    "tolerance": "webdisco_tolerance",
    "l2": "webdisco_l2",
    "ridge": "webdisco_ridge",
    "max_step_norm": "webdisco_max_step_norm",
}
PROTOCOL_FIELDS = frozenset(PROTOCOL_PARAMETER_ALIASES.values())
DISPLAY_METHODS = {
    "Center": "Center",
    "FSA": "Federated",
    "Local-client": "Local-client",
    "Local": "Weighted Local",
}
PLOT_TYPES = ("dotplot", "bar", "table", "boxplot", "violin")


def _validate_dataset(dataset: DataSet) -> Tuple[int, int, int]:
    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be the result of partition_data(...)")
    train = np.asarray(dataset.train_data)
    test = np.asarray(dataset.test_data)
    train_labels = np.asarray(dataset.train_label)
    test_labels = np.asarray(dataset.test_label)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] == 0:
        raise ValueError("training and test features must be non-empty two-dimensional arrays")
    if train.shape[1] != test.shape[1]:
        raise ValueError("training and test feature counts must match")
    if (
        train_labels.ndim != 2
        or test_labels.ndim != 2
        or train_labels.shape[1] != 2
        or test_labels.shape[1] != 2
        or len(train_labels) != len(train)
        or len(test_labels) != len(test)
    ):
        raise ValueError("training and test labels must have time/status columns")
    if not dataset.clients_set:
        raise ValueError("dataset must contain at least one client")
    if sum(len(features) for features, _ in dataset.clients_set.values()) != len(train):
        raise ValueError("client sample counts must equal the pooled training sample count")
    for client, (features, labels) in dataset.clients_set.items():
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != train.shape[1]:
            raise ValueError("client %s has an incompatible feature matrix" % client)
        if labels.ndim != 2 or labels.shape != (len(features), 2):
            raise ValueError("client %s must have matching time/status labels" % client)
        if len(features) == 0 or labels[:, 1].sum() <= 0:
            raise ValueError("client %s must contain an observed event" % client)
    return len(train) + len(test), train.shape[1], len(dataset.clients_set)


def _make_config(
    *,
    model: str,
    protocol: str,
    n_samples: int,
    n_features: int,
    n_clients: int,
    seed: int,
    global_rounds: int,
    local_steps: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    hidden_nodes: Optional[Sequence[int]],
    num_durations: int,
    activation: str,
    dropout: float,
    weight_decay: float,
    client_fraction: float,
    protocol_params: Optional[Mapping[str, object]],
) -> FSAConfig:
    if model not in MODEL_NAMES:
        raise ValueError("paired comparisons support these models: %s" % (list(MODEL_NAMES),))
    activation_name = str(activation).lower()
    if activation_name not in ACTIVATIONS:
        raise ValueError("activation must be one of %s" % list(ACTIVATIONS))
    if hidden_nodes is None:
        nodes = () if model == "CoxPH" else (32, 32)
    else:
        nodes = tuple(int(node) for node in hidden_nodes)
        if any(node <= 0 for node in nodes):
            raise ValueError("hidden_nodes must contain positive integers")
    if model == "CoxPH" and nodes:
        raise ValueError("CoxPH requires hidden_nodes=()")

    values = {
        "n_samples": n_samples,
        "n_features": n_features,
        "num_clients": n_clients,
        "model_type": model,
        "num_nodes": nodes,
        "num_durations": int(num_durations),
        "activation": ACTIVATIONS[activation_name],
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "global_epochs": int(global_rounds),
        "local_epochs": int(local_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "optimizer": optimizer,
        "client_sample_ratio": float(client_fraction),
        "federated_protocol": protocol,
        "random_seed": int(seed),
        "early_stopping": False,
        "prediction_validation": "raise",
        "show_progress": False,
        "verbose": False,
    }
    if protocol_params is not None:
        if not isinstance(protocol_params, Mapping):
            raise TypeError("protocol_params must be a mapping")
        valid_config_fields = {item.name for item in fields(FSAConfig)}
        for key, value in protocol_params.items():
            canonical = PROTOCOL_PARAMETER_ALIASES.get(str(key), str(key))
            if canonical not in PROTOCOL_FIELDS or canonical not in valid_config_fields:
                raise ValueError("unknown protocol parameter: %s" % key)
            values[canonical] = value
    return FSAConfig(**values)


@dataclass
class ExperimentResult:
    """Paired metrics, round history, and selectable plots from one or more seeds."""

    dataset: DataSet
    datasets: Dict[int, DataSet]
    metrics: pd.DataFrame
    round_metrics: pd.DataFrame
    predictions: pd.DataFrame
    configurations: Dict[Tuple[str, str], FSAConfig]
    _raw_metrics: pd.DataFrame

    def _selected_cell(self, model: Optional[str], protocol: Optional[str]) -> Tuple[str, str]:
        cells = list(self.configurations)
        if model is None and protocol is None and len(cells) == 1:
            return cells[0]
        if model is None or protocol is None:
            raise ValueError("select both model and protocol when plotting multiple cells")
        if (model, protocol) not in self.configurations:
            raise ValueError("no result for model=%r, protocol=%r" % (model, protocol))
        return model, protocol

    def plot(
        self,
        kind: str = "final_metrics",
        *,
        model: Optional[str] = None,
        protocol: Optional[str] = None,
        chart: Optional[str] = None,
        seed: Optional[int] = None,
        x: str = "feature_1",
        y: str = "time",
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """Plot client partition, final C-index/IBS, or federated round history.

        For multi-model/protocol results, select one cell with ``model`` and
        ``protocol``. Partition plots default to a client-wise strip; set
        ``chart="scatter"`` for feature axes. Boxplots and violin plots need
        at least two paired seeds.
        """
        if kind == "partition":
            selected_seed = next(iter(self.datasets)) if seed is None else int(seed)
            if selected_seed not in self.datasets:
                raise ValueError("no partition for seed %s" % selected_seed)
            return plot_partition(
                self.datasets[selected_seed],
                x=x,
                y=y,
                chart=chart or "strip",
                output_path=output_path,
                show=show,
            )
        if kind not in ("final_metrics", "round_metrics"):
            raise ValueError("kind must be partition, final_metrics, or round_metrics")
        model, protocol = self._selected_cell(model, protocol)

        import matplotlib.pyplot as plt

        from .experiments.visualization import _draw_final_metric, _draw_round_metric

        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
        if kind == "round_metrics":
            frame = self.round_metrics[
                self.round_metrics["model"].eq(model) & self.round_metrics["protocol"].eq(protocol)
            ]
            if frame.empty:
                plt.close(fig)
                raise ValueError("round metrics are unavailable for this result")
            for ax, metric in zip(axes, ("c_index", "ibs")):
                _draw_round_metric(ax, frame, metric, "mean_ci95")
        else:
            final_chart = chart or "dotplot"
            if final_chart not in PLOT_TYPES:
                plt.close(fig)
                raise ValueError("chart must be one of %s" % (list(PLOT_TYPES),))
            if final_chart in ("boxplot", "violin") and self._raw_metrics["seed"].nunique() < 2:
                plt.close(fig)
                raise ValueError("%s requires at least two paired seeds" % final_chart)
            frame = self._raw_metrics[
                self._raw_metrics["model"].eq(model) & self._raw_metrics["protocol"].eq(protocol)
            ]
            settings = {
                "type": final_chart,
                "methods": ["Center", "Federated", "Weighted Local"],
                "show_raw_points": True,
                "show_mean": True,
            }
            for ax, metric in zip(axes, ("c_index", "ibs")):
                _draw_final_metric(ax, frame, metric, settings)
        fig.tight_layout()
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def save_figures(
        self,
        output_dir: Union[str, Path],
        *,
        combine: bool = False,
        final_chart: str = "dotplot",
        partition_chart: str = "strip",
        model: Optional[str] = None,
        protocol: Optional[str] = None,
        seed: Optional[int] = None,
        x: str = "feature_1",
        y: str = "time",
        image_format: str = "png",
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """Save separate plots and optionally a dataset-independent combined figure.

        ``combine=False`` writes partition, round-metric, and final-metric
        figures. ``combine=True`` writes those plus one five-panel overview.
        Existing target files are preserved unless ``overwrite=True``.
        """
        model, protocol = self._selected_cell(model, protocol)
        selected_seed = next(iter(self.datasets)) if seed is None else int(seed)
        if selected_seed not in self.datasets:
            raise ValueError("no partition for seed %s" % selected_seed)
        if partition_chart not in ("strip", "scatter"):
            raise ValueError("partition_chart must be strip or scatter")
        if final_chart not in PLOT_TYPES:
            raise ValueError("final_chart must be one of %s" % (list(PLOT_TYPES),))
        if final_chart in ("boxplot", "violin") and len(self.datasets) < 2:
            raise ValueError("%s requires at least two paired seeds" % final_chart)
        if image_format not in ("png", "pdf", "svg"):
            raise ValueError("image_format must be png, pdf, or svg")
        if self.round_metrics.empty:
            raise ValueError("round metrics are unavailable for this result")
        rounds = self.round_metrics[
            self.round_metrics["model"].eq(model) & self.round_metrics["protocol"].eq(protocol)
        ]
        if rounds.empty:
            raise ValueError("round metrics are unavailable for this result")
        raw = self._raw_metrics[
            self._raw_metrics["model"].eq(model) & self._raw_metrics["protocol"].eq(protocol)
        ]
        directory = Path(output_dir)
        names = {
            "partition": "client_partition.%s" % image_format,
            "round_metrics": "round_metrics.%s" % image_format,
            "final_metrics": "final_metrics_%s.%s" % (final_chart, image_format),
        }
        if combine:
            names["combined"] = "combined_%s.%s" % (final_chart, image_format)
        paths = {key: directory / name for key, name in names.items()}
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError("figure files already exist: %s" % existing)
        directory.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        for key, options in (
            (
                "partition",
                {"chart": partition_chart, "seed": selected_seed, "x": x, "y": y},
            ),
            ("round_metrics", {"model": model, "protocol": protocol}),
            (
                "final_metrics",
                {"chart": final_chart, "model": model, "protocol": protocol},
            ),
        ):
            figure = self.plot(kind=key, output_path=paths[key], show=False, **options)
            plt.close(figure)

        if combine:
            from .experiments.visualization import _draw_final_metric, _draw_round_metric

            figure = plt.figure(figsize=(12.5, 13.5), constrained_layout=True)
            grid = figure.add_gridspec(3, 2, height_ratios=(1.25, 1, 1))
            partition_axis = figure.add_subplot(grid[0, :])
            _draw_partition(
                partition_axis,
                self.datasets[selected_seed],
                x,
                y,
                partition_chart,
            )
            round_axes = (figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]))
            final_axes = (figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1]))
            for axis, metric in zip(round_axes, ("c_index", "ibs")):
                _draw_round_metric(axis, rounds, metric, "mean_ci95")
            settings = {
                "type": final_chart,
                "methods": ["Center", "Federated", "Weighted Local"],
                "show_raw_points": True,
                "show_mean": True,
            }
            for axis, metric in zip(final_axes, ("c_index", "ibs")):
                _draw_final_metric(axis, raw, metric, settings)
            figure.savefig(paths["combined"], dpi=300, bbox_inches="tight")
            plt.close(figure)
        return paths


def compare_experiments(
    dataset: Union[DataSet, Mapping[int, DataSet]],
    *,
    models: Sequence[str],
    protocols: Sequence[str],
    global_rounds: int = 10,
    local_steps: int = 5,
    reference_steps: Optional[int] = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    hidden_nodes: Optional[Sequence[int]] = None,
    num_durations: int = 25,
    activation: str = "relu",
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    client_fraction: float = 1.0,
    seed: int = 42,
    protocol_params: Optional[Mapping[str, Mapping[str, object]]] = None,
    save_predictions: bool = False,
) -> ExperimentResult:
    """Cross selected models and protocols on paired per-seed partitions.

    Pass one ``DataSet`` for a single seed or a mapping returned by
    ``partition_data_many`` for repeated seeds. Within each seed, Center,
    Federated, and Local fits use the same partition and test set. Incompatible
    cells are rejected before training. This API does not write an audit
    manifest; use YAML when a full machine-readable run record is required.
    """
    if isinstance(dataset, DataSet):
        datasets = {int(seed): dataset}
    elif isinstance(dataset, Mapping) and dataset:
        datasets = {int(key): value for key, value in dataset.items()}
        if len(datasets) != len(dataset):
            raise ValueError("dataset seed keys must be unique integers")
    else:
        raise TypeError("dataset must be a DataSet or a non-empty seed-to-DataSet mapping")
    dimensions = {
        current_seed: _validate_dataset(split) for current_seed, split in datasets.items()
    }
    if len({(value[1], value[2]) for value in dimensions.values()}) != 1:
        raise ValueError("all seeds must have the same feature and client counts")
    if isinstance(models, str) or not models or len(set(models)) != len(models):
        raise ValueError("models must be a non-empty sequence without duplicates")
    if isinstance(protocols, str) or not protocols or len(set(protocols)) != len(protocols):
        raise ValueError("protocols must be a non-empty sequence without duplicates")
    if reference_steps is not None and int(reference_steps) <= 0:
        raise ValueError("reference_steps must be positive")
    parameter_map = protocol_params or {}
    if not isinstance(parameter_map, Mapping):
        raise TypeError("protocol_params must map protocol names to parameter mappings")

    configurations: Dict[Tuple[str, str], FSAConfig] = {}
    seeded_configurations: Dict[Tuple[int, str, str], FSAConfig] = {}
    for current_seed, dimensions_for_seed in dimensions.items():
        n_samples, n_features, n_clients = dimensions_for_seed
        for model in models:
            for requested_protocol in protocols:
                config = _make_config(
                    model=model,
                    protocol=requested_protocol,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_clients=n_clients,
                    seed=current_seed,
                    global_rounds=global_rounds,
                    local_steps=local_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    hidden_nodes=hidden_nodes,
                    num_durations=num_durations,
                    activation=activation,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    client_fraction=client_fraction,
                    protocol_params=parameter_map.get(requested_protocol),
                )
                cell_key = (model, config.federated_protocol)
                seeded_key = (current_seed,) + cell_key
                if seeded_key in seeded_configurations:
                    raise ValueError("duplicate model/protocol cell: %s" % (cell_key,))
                seeded_configurations[seeded_key] = config
                configurations[cell_key] = config

    raw_frames = []
    round_rows = []
    prediction_rows = [] if save_predictions else None
    for (current_seed, model, protocol), config in seeded_configurations.items():
        cell_rounds = []
        cell_predictions = [] if save_predictions else None
        cell = run_paired_baselines(
            config,
            datasets[current_seed],
            current_seed,
            baseline_epochs=reference_steps,
            telemetry_rows=cell_rounds,
            prediction_rows=cell_predictions,
        )
        cell["protocol"] = protocol
        raw_frames.append(cell)
        for row in cell_rounds:
            row["protocol"] = protocol
            round_rows.append(row)
        if cell_predictions is not None:
            for row in cell_predictions:
                row["protocol"] = protocol
                prediction_rows.append(row)

    raw = pd.concat(raw_frames, ignore_index=True)
    metrics = raw.copy()
    metrics["method"] = metrics["method"].map(DISPLAY_METHODS)
    return ExperimentResult(
        dataset=next(iter(datasets.values())),
        datasets=datasets,
        metrics=metrics,
        round_metrics=pd.DataFrame(round_rows),
        predictions=pd.DataFrame(prediction_rows or []),
        configurations=configurations,
        _raw_metrics=raw,
    )


def compare_methods(
    dataset: Union[DataSet, Mapping[int, DataSet]],
    *,
    model: str = "DeepSurv",
    protocol: str = "FedAvg",
    protocol_params: Optional[Mapping[str, object]] = None,
    **training_options,
) -> ExperimentResult:
    """Run one Center/Federated/Local comparison with a compact public call."""
    return compare_experiments(
        dataset,
        models=[model],
        protocols=[protocol],
        protocol_params={protocol: protocol_params or {}},
        **training_options,
    )


__all__ = ["ExperimentResult", "compare_methods", "compare_experiments"]
