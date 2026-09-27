"""Reusable experiment recording, configuration, and result validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


MANIFEST_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> Dict[str, Any]:
    """Return the minimal environment needed to audit a result."""
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "federated-survival",
                "numpy",
                "pandas",
                "scipy",
                "torch",
                "pycox",
                "torchtuples",
                "lifelines",
            )
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so interrupted runs do not leave valid-looking files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_experiment_config(path: Path) -> Dict[str, Any]:
    """Load JSON or YAML while rejecting non-mapping top-level values."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configuration requires the PyYAML dependency") from error
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("experiment configuration must use .json, .yaml, or .yml")
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must contain a mapping at the top level")
    return value


@dataclass
class ArtifactRecord:
    path: str
    kind: str
    sha256: str


@dataclass
class RunManifest:
    name: str
    configuration: Dict[str, Any]
    schema_version: str = MANIFEST_SCHEMA_VERSION
    status: str = "running"
    started_at: str = field(default_factory=_utc_now)
    completed_at: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=environment_report)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    error: Optional[str] = None


class ExperimentRecorder:
    """Create an auditable directory for one experiment invocation."""

    def __init__(self, output_dir: Path, name: str, configuration: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.manifest = RunManifest(name=name, configuration=configuration)
        self._flush()

    def add_artifact(self, path: Path, kind: str) -> ArtifactRecord:
        path = Path(path)
        record = ArtifactRecord(
            path=str(path.resolve()),
            kind=kind,
            sha256=sha256_file(path),
        )
        self.manifest.artifacts.append(record)
        self._flush()
        return record

    def write_metrics(self, frame: pd.DataFrame, filename: str = "metrics.csv") -> Path:
        validate_metrics(frame)
        destination = self.output_dir / filename
        frame.to_csv(destination, index=False)
        self.add_artifact(destination, "metrics")
        return destination

    def finish(self, status: str = "complete", error: Optional[str] = None) -> None:
        if status not in {"complete", "partial", "failed"}:
            raise ValueError("status must be complete, partial, or failed")
        self.manifest.status = status
        self.manifest.completed_at = _utc_now()
        self.manifest.error = error
        self._flush()

    def _flush(self) -> None:
        write_json(self.manifest_path, asdict(self.manifest))


def validate_metrics(frame: pd.DataFrame) -> None:
    """Reject incomplete or non-finite metric exports."""
    if frame.empty:
        raise ValueError("metrics table cannot be empty")
    metric_columns = [name for name in ("c_index", "ibs") if name in frame.columns]
    if not metric_columns:
        raise ValueError("metrics table must contain c_index or ibs")
    for name in metric_columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"metrics column {name!r} contains NaN or infinity")
    if "c_index" in frame and not frame["c_index"].between(0.0, 1.0).all():
        raise ValueError("c_index must lie between zero and one")
    if "ibs" in frame and (frame["ibs"] < 0.0).any():
        raise ValueError("ibs cannot be negative")

