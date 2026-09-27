#!/usr/bin/env python3
"""Run a small paired seven-model utility validation.

This is a software-validation experiment, not a publication-scale confirmatory
study.  It checks that all models and heterogeneity paths can be compared on
identical data with Center, FSA, and Local baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated_survival.core.config import FSAConfig
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.experiments.baselines import (
    MODEL_NAMES,
    run_paired_baselines,
    summarize_results,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/utility_smoke"))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["iid", "censoring-non-iid", "time-non-iid"],
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []

    for seed in range(args.seeds):
        frame = DataGenerator(
            SimulationConfig(args.samples, args.features, seed)
        ).generate("SDGM1", c_mean=0.4)
        for split in args.splits:
            dataset = DataSplitter(
                args.clients,
                split,
                test_size=0.2,
                random_state=seed,
            ).split(frame)
            for model in args.models:
                config = FSAConfig(
                    n_samples=args.samples,
                    n_features=args.features,
                    num_clients=args.clients,
                    model_type=model,
                    num_nodes=() if model == "CoxPH" else (32, 32),
                    num_durations=15,
                    global_epochs=args.rounds,
                    local_epochs=args.local_epochs,
                    learning_rate=args.learning_rate,
                    optimizer="adam",
                    weight_decay=0.0,
                    dropout=0.0,
                    full_batch=True,
                    client_sample_ratio=1.0,
                    split_method=split,
                    random_seed=seed,
                    early_stopping=False,
                    show_progress=False,
                )
                result = run_paired_baselines(config, dataset, seed)
                result["split"] = split
                frames.append(result)
                pd.concat(frames, ignore_index=True).to_csv(
                    args.output / "raw_results.csv", index=False
                )
                print(f"completed seed={seed} split={split} model={model}", flush=True)

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize_results(raw)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "scope": "software validation smoke experiment; not confirmatory evidence",
        "paired_design": True,
        "models": args.models,
        "splits": args.splits,
        "seeds": args.seeds,
        "samples": args.samples,
        "features": args.features,
        "clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "dropout": 0.0,
        "full_batch": True,
        "cox_loss_normalization": "patient",
        "privacy": False,
        "augmentation": False,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
