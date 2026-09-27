"""Create manuscript and supplementary tables for software coverage validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/software_validation")
MODEL_DIR = ROOT / "seven_model_validation_5seeds"
PROTOCOL_DIR = ROOT / "four_protocol_validation_5seeds"
OUTPUT_DIR = ROOT / "tables"


def _read_complete_run(directory: Path) -> tuple[pd.DataFrame, dict]:
    metrics = pd.read_csv(directory / "metrics.csv")
    failures = pd.read_csv(directory / "failures.csv")
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete: {directory}")
    if not failures.empty:
        raise RuntimeError(f"run contains failures: {directory}")
    values = metrics[["c_index", "ibs"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"run contains non-finite metrics: {directory}")
    return metrics, manifest


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(groups, sort=False)
        .agg(
            repetitions=("seed", "nunique"),
            c_index_mean=("c_index", "mean"),
            c_index_sd=("c_index", "std"),
            ibs_mean=("ibs", "mean"),
            ibs_sd=("ibs", "std"),
        )
        .reset_index()
    )
    numeric = ["c_index_mean", "c_index_sd", "ibs_mean", "ibs_sd"]
    summary[numeric] = summary[numeric].round(4)
    return summary


def main() -> None:
    model_metrics, model_manifest = _read_complete_run(MODEL_DIR)
    protocol_metrics, protocol_manifest = _read_complete_run(PROTOCOL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coverage_rows = []
    for model, group in model_metrics.groupby("model", sort=False):
        repetitions = int(group["seed"].nunique())
        coverage_rows.append(
            {
                "component_type": "Model adapter",
                "component": model,
                "validation_setting": "FedAvg",
                "completed_runs": f"{len(group)}/{repetitions}",
                "valid_survival_predictions": "Yes",
                "finite_C_index_and_IBS": "Yes",
            }
        )

    for protocol, group in protocol_metrics.groupby("protocol", sort=False):
        model_count = int(group["model"].nunique())
        seed_count = int(group["seed"].nunique())
        setting = "CoxPH" if protocol == "WebDISCO-style" else f"{model_count} model adapters"
        expected = model_count * seed_count
        coverage_rows.append(
            {
                "component_type": "Protocol adapter",
                "component": protocol,
                "validation_setting": setting,
                "completed_runs": f"{len(group)}/{expected}",
                "valid_survival_predictions": "Yes",
                "finite_C_index_and_IBS": "Yes",
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT_DIR / "main_coverage_table.csv", index=False)
    model_rows = coverage[coverage["component_type"] == "Model adapter"]
    protocol_rows = coverage[coverage["component_type"] == "Protocol adapter"]
    latex_lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Coverage validation of the built-in model and protocol adapters.}",
        r"\label{tab:software-coverage}",
        r"\small",
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Component & Validation setting & Completed & Valid prediction & Finite metrics \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Model adapters}} \\",
    ]
    for row in model_rows.to_dict("records"):
        component = row["component"].replace("_", r"\_")
        setting = row["validation_setting"].replace("_", r"\_")
        latex_lines.append(
            f"{component} & {setting} & {row['completed_runs']} & Yes & Yes " + r"\\"
        )
    latex_lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{Protocol adapters}} \\",
        ]
    )
    for row in protocol_rows.to_dict("records"):
        component = row["component"].replace("_", r"\_")
        setting = row["validation_setting"].replace("_", r"\_")
        latex_lines.append(
            f"{component} & {setting} & {row['completed_runs']} & Yes & Yes " + r"\\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\parbox{\linewidth}{\footnotesize Each validation cell was repeated with five random seeds. "
            r"Prediction validation required finite values, probabilities within $[0,1]$, and survival "
            r"probabilities that were non-increasing over time. Metrics were required to contain finite "
            r"C-index and IBS values. These runs assess software coverage and repeatability rather than "
            r"comparative performance.}",
            r"\end{table}",
        ]
    )
    (OUTPUT_DIR / "main_coverage_table.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )

    model_summary = _summary(model_metrics, ["protocol", "model"])
    protocol_summary = _summary(protocol_metrics, ["protocol", "model"])
    model_summary.to_csv(OUTPUT_DIR / "supplement_model_metrics_summary.csv", index=False)
    protocol_summary.to_csv(
        OUTPUT_DIR / "supplement_protocol_model_metrics_summary.csv", index=False
    )

    report = {
        "model_validation": {
            "status": model_manifest["status"],
            "models": int(model_metrics["model"].nunique()),
            "seeds": sorted(int(value) for value in model_metrics["seed"].unique()),
            "completed_cells": int(len(model_metrics)),
            "failed_cells": 0,
            "non_finite_metric_rows": 0,
        },
        "protocol_validation": {
            "status": protocol_manifest["status"],
            "protocols": int(protocol_metrics["protocol"].nunique()),
            "model_protocol_combinations": int(
                protocol_metrics[["protocol", "model"]].drop_duplicates().shape[0]
            ),
            "seeds": sorted(int(value) for value in protocol_metrics["seed"].unique()),
            "completed_cells": int(len(protocol_metrics)),
            "failed_cells": 0,
            "non_finite_metric_rows": 0,
        },
        "interpretation": (
            "Software coverage and repeatability check; metrics are descriptive and are not "
            "intended as a performance ranking."
        ),
    }
    (OUTPUT_DIR / "supplement_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
