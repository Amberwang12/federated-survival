"""Schema normalization for configuration-driven end-to-end experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

DATA_SOURCES = ("csv", "excel", "simulation")
WORKFLOW_STAGES = ("data_only", "full")
MISSING_VALUE_POLICIES = ("error", "drop", "median")
PARTITION_METHODS = (
    "iid",
    "random",
    "censoring-non-iid",
    "time-non-iid",
    "dirichlet",
)
MODEL_NAMES = (
    "CoxPH",
    "DeepSurv",
    "CoxCC",
    "CoxTime",
    "LogisticHazard",
    "PC-Hazard",
    "DeepHit",
)
PROTOCOL_NAMES = ("FedAvg", "FedProx", "FedOpt", "WebDISCO-style")
METRIC_NAMES = ("c_index", "ibs")
FINAL_PLOT_TYPES = ("boxplot", "violin", "dotplot", "bar", "table")
PARTITION_PLOT_TYPES = ("strip", "scatter", "boxplot", "violin", "censoring_bar")
OUTPUT_FORMATS = ("png", "pdf", "svg")


def is_workflow_configuration(configuration: Dict[str, Any]) -> bool:
    """Return whether a mapping uses the end-to-end workflow schema."""
    return any(
        key in configuration
        for key in ("schema_version", "experiment", "partition", "model", "visualization")
    )


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping" % name)
    return deepcopy(value)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("%s must be positive" % name)
    return parsed


def _positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    parsed = float(value)
    if parsed < 0 if allow_zero else parsed <= 0:
        condition = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be %s" % (name, condition))
    return parsed


def _choice(value: Any, choices: Iterable[str], name: str) -> str:
    rendered = str(value)
    if rendered not in choices:
        raise ValueError("%s must be one of %s" % (name, list(choices)))
    return rendered


def normalize_workflow_config(configuration: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Apply defaults, resolve input paths, and reject invalid combinations."""
    normalized = deepcopy(configuration)
    normalized["schema_version"] = int(normalized.get("schema_version", 1))
    if normalized["schema_version"] != 1:
        raise ValueError("only schema_version 1 is supported")
    normalized["name"] = str(normalized.get("name", config_path.stem))

    experiment = _mapping(normalized.get("experiment"), "experiment")
    seeds = experiment.get("seeds", normalized.get("seeds", [0]))
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    parsed_seeds: List[int] = [int(seed) for seed in seeds]
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("experiment.seeds must not contain duplicates")
    experiment["seeds"] = parsed_seeds
    experiment["fail_fast"] = bool(experiment.get("fail_fast", False))
    experiment["stage"] = _choice(
        experiment.get("stage", "full"), WORKFLOW_STAGES, "experiment.stage"
    )
    normalized["experiment"] = experiment

    data = _mapping(normalized.get("data"), "data")
    source = _choice(data.get("source", "simulation"), DATA_SOURCES, "data.source")
    data["source"] = source
    data["duration_column"] = str(data.get("duration_column", "time"))
    data["event_column"] = str(data.get("event_column", "status"))
    data["test_size"] = float(data.get("test_size", 0.2))
    if not 0 < data["test_size"] < 1:
        raise ValueError("data.test_size must be between zero and one")
    data["standardize"] = bool(data.get("standardize", source != "simulation"))
    data["missing_values"] = _choice(
        data.get("missing_values", "error"),
        MISSING_VALUE_POLICIES,
        "data.missing_values",
    )
    if source in ("csv", "excel"):
        if not data.get("path"):
            raise ValueError("data.path is required for CSV or Excel input")
        path = Path(str(data["path"]))
        if not path.is_absolute():
            path = config_path.parent / path
        data["resolved_path"] = str(path.resolve())
        if not path.exists():
            raise FileNotFoundError("data file not found: %s" % path)
    else:
        data["mechanism"] = str(data.get("mechanism", "weibull"))
        data["n_samples"] = _positive_int(data.get("n_samples", 300), "data.n_samples")
        data["n_features"] = _positive_int(data.get("n_features", 10), "data.n_features")
        data["censoring_rate"] = float(data.get("censoring_rate", 0.4))
        if not 0 <= data["censoring_rate"] <= 1:
            raise ValueError("data.censoring_rate must be between zero and one")
    normalized["data"] = data

    partition = _mapping(normalized.get("partition"), "partition")
    partition["method"] = _choice(
        str(partition.get("method", "iid")).lower(),
        PARTITION_METHODS,
        "partition.method",
    )
    partition["n_clients"] = _positive_int(partition.get("n_clients", 5), "partition.n_clients")
    partition["stratify_by"] = str(partition.get("stratify_by", "status"))
    if partition["stratify_by"] != "status":
        raise ValueError("schema version 1 supports partition.stratify_by: status")
    partition["alpha"] = _positive_float(partition.get("alpha", 0.5), "partition.alpha")
    normalized["partition"] = partition

    model = _mapping(normalized.get("model"), "model")
    model["name"] = _choice(model.get("name", "DeepSurv"), MODEL_NAMES, "model.name")
    hidden = model.get("hidden_nodes", [] if model["name"] == "CoxPH" else [32, 32])
    if not isinstance(hidden, list) or any(int(node) <= 0 for node in hidden):
        raise ValueError("model.hidden_nodes must be a list of positive integers")
    if model["name"] == "CoxPH" and hidden:
        raise ValueError("CoxPH requires model.hidden_nodes: []")
    model["hidden_nodes"] = [int(node) for node in hidden]
    model["activation"] = _choice(
        str(model.get("activation", "relu")).lower(),
        ("relu", "tanh", "softplus", "sigmoid"),
        "model.activation",
    )
    model["dropout"] = float(model.get("dropout", 0.1))
    if not 0 <= model["dropout"] <= 1:
        raise ValueError("model.dropout must be between zero and one")
    model["weight_decay"] = _positive_float(
        model.get("weight_decay", 0.0), "model.weight_decay", allow_zero=True
    )
    model["num_durations"] = _positive_int(model.get("num_durations", 25), "model.num_durations")
    normalized["model"] = model

    federated = _mapping(normalized.get("federated"), "federated")
    federated["protocol"] = _choice(
        federated.get("protocol", "FedAvg"), PROTOCOL_NAMES, "federated.protocol"
    )
    federated["global_rounds"] = _positive_int(
        federated.get("global_rounds", 10), "federated.global_rounds"
    )
    federated["local_steps"] = _positive_int(
        federated.get("local_steps", 1), "federated.local_steps"
    )
    federated["client_fraction"] = float(federated.get("client_fraction", 1.0))
    if not 0 < federated["client_fraction"] <= 1:
        raise ValueError("federated.client_fraction must be in (0, 1]")
    federated["batch_mode"] = _choice(
        federated.get("batch_mode", "mini_batch"),
        ("mini_batch", "full_batch"),
        "federated.batch_mode",
    )
    federated["batch_size"] = _positive_int(federated.get("batch_size", 32), "federated.batch_size")
    federated["optimizer"] = _choice(
        str(federated.get("optimizer", "adam")).lower(),
        ("adam", "sgd"),
        "federated.optimizer",
    )
    federated["learning_rate"] = _positive_float(
        federated.get("learning_rate", 1e-3), "federated.learning_rate"
    )
    federated["protocol_params"] = _mapping(
        federated.get("protocol_params"), "federated.protocol_params"
    )
    normalized["federated"] = federated

    references = _mapping(normalized.get("references"), "references")
    references["centralized"] = bool(references.get("centralized", True))
    references["local_models"] = bool(references.get("local_models", True))
    references["aggregate_local"] = _choice(
        references.get("aggregate_local", "sample_weighted"),
        ("sample_weighted",),
        "references.aggregate_local",
    )
    default_steps = federated["global_rounds"] * federated["local_steps"]
    references["optimizer_steps"] = _positive_int(
        references.get("optimizer_steps", default_steps),
        "references.optimizer_steps",
    )
    normalized["references"] = references

    evaluation = _mapping(normalized.get("evaluation"), "evaluation")
    metrics = evaluation.get("metrics", list(METRIC_NAMES))
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("evaluation.metrics must be a non-empty list")
    unknown_metrics = sorted(set(metrics) - set(METRIC_NAMES))
    if unknown_metrics:
        raise ValueError("unsupported evaluation metrics: %s" % unknown_metrics)
    evaluation["metrics"] = metrics
    evaluation["save_round_metrics"] = bool(evaluation.get("save_round_metrics", True))
    evaluation["save_predictions"] = bool(evaluation.get("save_predictions", False))
    quantiles = evaluation.get("time_grid_quantiles", [0.05, 0.95])
    if not isinstance(quantiles, list) or len(quantiles) != 2:
        raise ValueError("evaluation.time_grid_quantiles must contain two values")
    quantiles = [float(value) for value in quantiles]
    if not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("evaluation.time_grid_quantiles must satisfy 0 <= low < high <= 1")
    evaluation["time_grid_quantiles"] = quantiles
    normalized["evaluation"] = evaluation

    visualization = _mapping(normalized.get("visualization"), "visualization")
    visualization["enabled"] = bool(visualization.get("enabled", False))
    visualization["style"] = _choice(
        visualization.get("style", "publication"),
        ("publication", "presentation"),
        "visualization.style",
    )
    formats = visualization.get("formats", ["png", "pdf"])
    if not isinstance(formats, list) or not formats:
        raise ValueError("visualization.formats must be a non-empty list")
    unknown_formats = sorted(set(formats) - set(OUTPUT_FORMATS))
    if unknown_formats:
        raise ValueError("unsupported visualization formats: %s" % unknown_formats)
    visualization["formats"] = formats

    partition_plot = _mapping(
        visualization.get("client_partition"), "visualization.client_partition"
    )
    partition_plot["type"] = _choice(
        partition_plot.get("type", "strip"),
        PARTITION_PLOT_TYPES,
        "visualization.client_partition.type",
    )
    partition_plot["show_sample_size"] = bool(partition_plot.get("show_sample_size", True))
    partition_plot["show_censoring_rate"] = bool(partition_plot.get("show_censoring_rate", True))
    partition_plot["x"] = str(partition_plot.get("x", "feature_1"))
    partition_plot["y"] = str(partition_plot.get("y", "time"))
    for axis in ("x", "y"):
        value = partition_plot[axis]
        valid_feature = value.startswith("feature_") and value[8:].isdigit() and int(value[8:]) > 0
        if value != "time" and not valid_feature:
            raise ValueError("visualization.client_partition.%s must be time or feature_N" % axis)
    visualization["client_partition"] = partition_plot

    round_plot = _mapping(visualization.get("round_metrics"), "visualization.round_metrics")
    round_plot["type"] = _choice(
        round_plot.get("type", "line"), ("line",), "visualization.round_metrics.type"
    )
    round_plot["aggregate_seeds"] = _choice(
        round_plot.get("aggregate_seeds", "mean_ci95"),
        ("none", "mean_sd", "mean_ci95"),
        "visualization.round_metrics.aggregate_seeds",
    )
    round_plot["separate_panels"] = bool(round_plot.get("separate_panels", True))
    if not round_plot["separate_panels"]:
        raise ValueError(
            "schema version 1 requires visualization.round_metrics.separate_panels: true"
        )
    visualization["round_metrics"] = round_plot

    final_plot = _mapping(visualization.get("final_metrics"), "visualization.final_metrics")
    final_plot["type"] = _choice(
        final_plot.get("type", "boxplot"),
        FINAL_PLOT_TYPES,
        "visualization.final_metrics.type",
    )
    requested_final_metrics = final_plot.get("metrics", evaluation["metrics"])
    if not isinstance(requested_final_metrics, list):
        raise ValueError("visualization.final_metrics.metrics must be a list")
    final_plot["metrics"] = list(requested_final_metrics)
    if not final_plot["metrics"] or set(final_plot["metrics"]) - set(METRIC_NAMES):
        raise ValueError("visualization.final_metrics.metrics supports c_index and ibs")
    requested_methods = final_plot.get("methods", ["Center", "Federated", "Weighted Local"])
    if not isinstance(requested_methods, list) or not requested_methods:
        raise ValueError("visualization.final_metrics.methods must be a non-empty list")
    allowed_methods = {"Center", "Federated", "Weighted Local"}
    unknown_methods = sorted(set(requested_methods) - allowed_methods)
    if unknown_methods:
        raise ValueError("unsupported visualization methods: %s" % unknown_methods)
    final_plot["methods"] = list(requested_methods)
    final_plot["group_by"] = str(final_plot.get("group_by", "method"))
    final_plot["replicate_by"] = str(final_plot.get("replicate_by", "seed"))
    if final_plot["group_by"] != "method" or final_plot["replicate_by"] != "seed":
        raise ValueError(
            "schema version 1 requires final_metrics group_by: method and replicate_by: seed"
        )
    final_plot["separate_panels"] = bool(final_plot.get("separate_panels", True))
    if not final_plot["separate_panels"]:
        raise ValueError(
            "schema version 1 requires visualization.final_metrics.separate_panels: true"
        )
    final_plot["show_raw_points"] = bool(final_plot.get("show_raw_points", True))
    final_plot["show_mean"] = bool(final_plot.get("show_mean", True))
    visualization["final_metrics"] = final_plot

    composite = _mapping(visualization.get("composite"), "visualization.composite")
    composite["enabled"] = bool(composite.get("enabled", False))
    composite["preset"] = _choice(
        composite.get("preset", "overview"),
        ("overview",),
        "visualization.composite.preset",
    )
    visualization["composite"] = composite
    if visualization["enabled"] and composite["enabled"] and not evaluation["save_round_metrics"]:
        raise ValueError("composite visualization requires evaluation.save_round_metrics: true")
    if (
        visualization["enabled"]
        and experiment["stage"] == "full"
        and final_plot["type"] in ("boxplot", "violin")
        and len(parsed_seeds) < 2
    ):
        raise ValueError("%s requires at least two experiment.seeds" % final_plot["type"])
    normalized["visualization"] = visualization

    output = _mapping(normalized.get("output"), "output")
    output["directory"] = str(output.get("directory", "results/%s" % normalized["name"]))
    output["overwrite"] = bool(output.get("overwrite", False))
    normalized["output"] = output
    return normalized
