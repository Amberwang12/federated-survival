"""Command-line interface for diagnostics and reproducible experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from . import __version__
from .api import FederatedSurvival
from .data import DataGenerator, SimulationConfig
from .experiments.config_runner import run_config_file
from .experiments.config_schema import (
    DATA_SOURCES,
    FINAL_PLOT_TYPES,
    METRIC_NAMES,
    MISSING_VALUE_POLICIES,
    OUTPUT_FORMATS,
    PARTITION_METHODS,
    PARTITION_PLOT_TYPES,
    PROTOCOL_NAMES,
    WORKFLOW_STAGES,
)
from .experiments.infrastructure import (
    ExperimentRecorder,
    environment_report,
    write_json,
)
from .models import available_model_adapters, get_model_adapter
from .protocols import available_federated_protocols, get_federated_protocol


def doctor_report() -> Dict[str, Any]:
    adapters = {}
    for name in available_model_adapters():
        try:
            adapters[name] = get_model_adapter(name).name == name
        except Exception:
            adapters[name] = False
    protocols = {}
    for name in available_federated_protocols():
        try:
            protocols[name] = get_federated_protocol(name).name == name
        except Exception:
            protocols[name] = False
    return {
        "status": ("passed" if all(adapters.values()) and all(protocols.values()) else "failed"),
        "package_version": __version__,
        "adapters": adapters,
        "protocols": protocols,
        "environment": environment_report(),
    }


def command_doctor(args) -> int:
    report = doctor_report()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        write_json(Path(args.output), report)
    return 0 if report["status"] == "passed" else 1


def command_quickstart(args) -> int:
    configuration = vars(args).copy()
    configuration.pop("handler", None)
    output_dir = Path(args.output)
    recorder = ExperimentRecorder(output_dir, "quickstart", configuration)
    try:
        frame = DataGenerator(
            SimulationConfig(
                n_samples=args.samples,
                n_features=args.features,
                random_state=args.seed,
            )
        ).generate(args.mechanism, c_mean=args.censoring)
        estimator = FederatedSurvival(
            model=args.model,
            protocol=args.protocol,
            protocol_params={
                "mu": args.proximal_mu,
                "optimizer": args.server_optimizer,
                "server_lr": args.server_learning_rate,
            },
            n_clients=args.clients,
            global_rounds=args.rounds,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            split_type=args.split_type,
            random_state=args.seed,
            num_nodes=() if args.model == "CoxPH" else (32, 32),
            num_durations=args.num_durations,
            weight_decay=0.0,
        ).fit(frame)
        metric_names = ("train_Cindex", "train_IBS", "test_Cindex", "test_IBS")
        history = pd.DataFrame({name: estimator.history_[name] for name in metric_names})
        history.insert(0, "round", range(1, len(history) + 1))
        history_path = output_dir / "round_metrics.csv"
        history.to_csv(history_path, index=False)
        recorder.add_artifact(history_path, "round_metrics")

        survival = estimator.predict_survival(estimator.dataset_.test_data)
        survival_path = output_dir / "survival_predictions.csv"
        survival.to_csv(survival_path, index_label="time")
        recorder.add_artifact(survival_path, "predictions")

        summary_path = output_dir / "summary.json"
        write_json(summary_path, estimator.get_run_summary())
        recorder.add_artifact(summary_path, "summary")
        recorder.finish("complete")
        print(f"Quickstart completed: {output_dir.resolve()}")
        return 0
    except Exception as error:
        recorder.finish("failed", f"{type(error).__name__}: {error}")
        raise


def command_run_config(args) -> int:
    output = run_config_file(Path(args.config), Path(args.output) if args.output else None)
    print(f"Configured benchmark completed: {output.resolve()}")
    return 0


def command_config_options(args) -> int:
    """Print choices accepted by schema-v1 configuration files."""
    report = {
        "data_sources": list(DATA_SOURCES),
        "workflow_stages": list(WORKFLOW_STAGES),
        "missing_value_policies": list(MISSING_VALUE_POLICIES),
        "partition_methods": list(PARTITION_METHODS),
        "models": list(available_model_adapters()),
        "protocols": list(PROTOCOL_NAMES),
        "metrics": list(METRIC_NAMES),
        "client_partition_plots": list(PARTITION_PLOT_TYPES),
        "final_metric_plots": list(FINAL_PLOT_TYPES),
        "figure_formats": list(OUTPUT_FORMATS),
        "data_preview_example": ("federated_survival/examples/simulation_partition_scatter.yaml"),
        "documentation": "docs/configuration.md",
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        write_json(Path(args.output), report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="federated-survival",
        description="Validate and run reproducible federated survival experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect the installation and model registry")
    doctor.add_argument("--output", help="optional JSON report path")
    doctor.set_defaults(handler=command_doctor)

    quickstart = subparsers.add_parser("quickstart", help="run one small audited example")
    quickstart.add_argument("--output", default="results/quickstart")
    quickstart.add_argument("--model", choices=available_model_adapters(), default="DeepSurv")
    quickstart.add_argument("--protocol", choices=available_federated_protocols(), default="FedAvg")
    quickstart.add_argument("--proximal-mu", type=float, default=0.01)
    quickstart.add_argument(
        "--server-optimizer", choices=["adam", "yogi", "adagrad"], default="adam"
    )
    quickstart.add_argument("--server-learning-rate", type=float, default=0.01)
    quickstart.add_argument("--mechanism", default="weibull")
    quickstart.add_argument("--samples", type=int, default=200)
    quickstart.add_argument("--features", type=int, default=5)
    quickstart.add_argument("--clients", type=int, default=3)
    quickstart.add_argument("--rounds", type=int, default=2)
    quickstart.add_argument("--local-steps", type=int, default=1)
    quickstart.add_argument("--batch-size", type=int, default=32)
    quickstart.add_argument("--num-durations", type=int, default=20)
    quickstart.add_argument("--split-type", default="iid")
    quickstart.add_argument("--censoring", type=float, default=0.4)
    quickstart.add_argument("--seed", type=int, default=42)
    quickstart.set_defaults(handler=command_quickstart)

    configured = subparsers.add_parser("run", help="run a JSON or YAML experiment")
    configured.add_argument("--config", required=True)
    configured.add_argument("--output")
    configured.set_defaults(handler=command_run_config)

    options = subparsers.add_parser(
        "config-options", help="list schema-v1 data, model, protocol, and plot choices"
    )
    options.add_argument("--output", help="optional JSON report path")
    options.set_defaults(handler=command_config_options)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
