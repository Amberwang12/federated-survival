"""High-level public API for federated survival analysis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split

from .core import FSAConfig, FSARunner
from .data.splitter import DataSet, DataSplitter
from .models import available_model_adapters
from .protocols import available_federated_protocols
from .utils.metrics import evaluation_time_grid


class FederatedSurvival:
    """A compact, model-independent entry point.

    Parameters use user-facing names while mapping onto the backward-compatible
    :class:`FSAConfig` fields.  Advanced users may pass an existing ``config``
    instead.
    """

    def __init__(
        self,
        model: str = "DeepSurv",
        *,
        protocol: str = "FedAvg",
        protocol_params: Optional[Dict[str, Any]] = None,
        config: Optional[FSAConfig] = None,
        n_clients: int = 5,
        global_rounds: int = 20,
        local_steps: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        client_fraction: float = 1.0,
        split_type: str = "iid",
        split_alpha: float = 0.5,
        test_size: float = 0.2,
        random_state: int = 42,
        **config_overrides,
    ) -> None:
        if config is not None and (config_overrides or protocol_params):
            raise ValueError(
                "protocol_params/config_overrides cannot be combined with an existing config"
            )
        self.split_type = split_type
        self.split_alpha = split_alpha
        self.test_size = test_size
        if config is None:
            values = {
                "model_type": model,
                "federated_protocol": protocol,
                "num_clients": n_clients,
                "global_epochs": global_rounds,
                "local_epochs": local_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "client_sample_ratio": client_fraction,
                "split_method": split_type,
                "split_alpha": split_alpha,
                "test_size": test_size,
                "random_seed": random_state,
                "prediction_validation": "raise",
            }
            if protocol_params:
                parameter_aliases = {
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
                for key, value in protocol_params.items():
                    values[parameter_aliases.get(key, key)] = value
            values.update(config_overrides)
            if model == "CoxPH" and "num_nodes" not in values:
                values["num_nodes"] = ()
            config = FSAConfig(**values)
        else:
            self.split_type = config.split_method
            self.split_alpha = config.split_alpha
            self.test_size = config.test_size
        self.config = config
        self.runner_: Optional[FSARunner] = None
        self.history_: Optional[Dict[str, Any]] = None
        self.feature_names_: Optional[Sequence[str]] = None
        self.dataset_: Optional[DataSet] = None

    @property
    def available_models(self):
        return available_model_adapters()

    @property
    def available_protocols(self):
        return available_federated_protocols()

    def fit(
        self,
        data,
        durations=None,
        events=None,
        *,
        duration_col: str = "time",
        event_col: str = "status",
        client_col: Optional[str] = None,
        augmentation: Optional[str] = None,
    ) -> "FederatedSurvival":
        """Fit from a package ``DataSet``, DataFrame, or NumPy arrays.

        A DataFrame may optionally contain an institution column.  Otherwise
        the requested synthetic partition strategy is applied.  The feature
        count is always inferred from the supplied data.
        """

        dataset = self._prepare_dataset(
            data,
            durations,
            events,
            duration_col=duration_col,
            event_col=event_col,
            client_col=client_col,
        )
        first_x = next(iter(dataset.clients_set.values()))[0]
        self.config.n_features = int(first_x.shape[1])
        self.config.num_clients = len(dataset.clients_set)

        runner = FSARunner(self.config)
        if augmentation is None:
            history = runner.run(dataset, type="raw")
        else:
            normalized = augmentation.upper()
            if normalized not in {"MVAEC", "MVAES"}:
                raise ValueError("augmentation must be None, 'MVAEC', or 'MVAES'")
            history = runner.run(dataset, type="raw_aug", aug_method=normalized)

        self.runner_ = runner
        self.history_ = history
        self.dataset_ = dataset
        return self

    def predict_survival(self, features) -> pd.DataFrame:
        """Return a time-by-sample survival-probability DataFrame."""
        self._require_fitted()
        array = self._coerce_features(features)
        return self.runner_.predict_survival(array)

    def evaluate(self, features, durations, events) -> Dict[str, Any]:
        """Compute time-dependent concordance and IPCW integrated Brier score."""
        survival = self.predict_survival(features)
        duration_values = np.asarray(durations, dtype=float).reshape(-1)
        event_values = np.asarray(events, dtype=float).reshape(-1)
        if survival.shape[1] != len(duration_values) or len(events) != len(duration_values):
            raise ValueError("features, durations, and events must contain the same samples")
        evaluator = EvalSurv(
            survival,
            duration_values,
            event_values,
            censor_surv="km",
        )
        grid = evaluation_time_grid(
            duration_values,
            survival.index.values,
            self.config.evaluation_quantiles,
        )
        diagnostics = self.runner_.last_prediction_diagnostics_
        return {
            "c_index": float(evaluator.concordance_td()),
            "ibs": float(evaluator.integrated_brier_score(grid)),
            "prediction_diagnostics": asdict(diagnostics),
        }

    def get_run_summary(self) -> Dict[str, Any]:
        """Return serialisable configuration and final metrics."""
        self._require_fitted()
        final = {}
        for name in ("train_Cindex", "train_IBS", "test_Cindex", "test_IBS"):
            values = self.history_.get(name, [])
            final[name] = float(values[-1]) if values else None
        return {
            "model": self.config.model_type,
            "protocol": self.config.federated_protocol,
            "n_clients": self.config.num_clients,
            "n_features": self.config.n_features,
            "global_rounds": self.config.global_epochs,
            "local_steps": self.config.local_epochs,
            "batch_size": self.config.batch_size,
            "final_metrics": final,
        }

    def get_privacy_info(self) -> Dict[str, Any]:
        """Report configured perturbation settings, not an achieved DP budget."""
        self._require_fitted()
        config = self.config
        if not config.use_differential_privacy:
            return {"enabled": False, "formal_accounting_available": False}
        return {
            "enabled": True,
            "mechanism": config.dp_mechanism,
            "configured_epsilon": config.dp_epsilon,
            "configured_delta": config.dp_delta if config.dp_mechanism == "gaussian" else None,
            "configured_sensitivity": config.dp_sensitivity,
            "configured_noise_multiplier": (
                config.dp_noise_multiplier if config.dp_mechanism == "gaussian" else None
            ),
            "configured_clip_norm": (
                config.dp_clip_norm if config.dp_mechanism in ("gaussian", "laplace") else None
            ),
            "formal_accounting_available": False,
            "privacy_scope": "experimental clipped client-update perturbation",
        }

    def _prepare_dataset(
        self,
        data,
        durations,
        events,
        *,
        duration_col,
        event_col,
        client_col,
    ) -> DataSet:
        if isinstance(data, DataSet):
            first_x = next(iter(data.clients_set.values()))[0]
            self.feature_names_ = tuple(f"x{index + 1}" for index in range(first_x.shape[1]))
            return data

        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            features = np.asarray(data)
            if features.ndim != 2:
                raise ValueError("feature data must be a two-dimensional array")
            if durations is None or events is None:
                raise ValueError("durations and events are required with array input")
            if len(features) != len(durations) or len(features) != len(events):
                raise ValueError("features, durations, and events must have equal length")
            names = [f"x{index + 1}" for index in range(features.shape[1])]
            frame = pd.DataFrame(features, columns=names)
            frame[duration_col] = np.asarray(durations)
            frame[event_col] = np.asarray(events)

        required = {duration_col, event_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"data is missing required columns: {sorted(missing)}")
        rename = {}
        if duration_col != "time":
            rename[duration_col] = "time"
        if event_col != "status":
            rename[event_col] = "status"
        frame = frame.rename(columns=rename)
        duration_col, event_col = "time", "status"

        excluded = {duration_col, event_col}
        if client_col is not None:
            if client_col not in frame:
                raise ValueError(f"client column {client_col!r} was not found")
            excluded.add(client_col)
        feature_names = [name for name in frame.columns if name not in excluded]
        if not feature_names:
            raise ValueError("at least one feature column is required")
        non_numeric = [
            name for name in feature_names if not pd.api.types.is_numeric_dtype(frame[name])
        ]
        if non_numeric:
            raise TypeError(f"feature columns must be numeric: {non_numeric}")
        self.feature_names_ = tuple(feature_names)

        canonical = frame[
            feature_names + [duration_col, event_col] + ([client_col] if client_col else [])
        ]
        if client_col is None:
            return DataSplitter(
                n_clients=self.config.num_clients,
                split_type=self.split_type,
                alpha=self.split_alpha,
                test_size=self.test_size,
                random_state=self.config.random_seed,
            ).split(canonical)
        return self._split_existing_clients(canonical, feature_names, client_col)

    def _split_existing_clients(self, frame, feature_names, client_col) -> DataSet:
        train, test = train_test_split(
            frame,
            test_size=self.test_size,
            random_state=self.config.random_seed,
            stratify=frame["status"],
        )
        clients = {}
        for index, (_, group) in enumerate(train.groupby(client_col, sort=True)):
            x = group[feature_names].to_numpy(dtype=np.float32)
            y = group[["time", "status"]].to_numpy(dtype=np.float32)
            if not len(x):
                continue
            if y[:, 1].sum() < 1:
                raise ValueError(f"client {index} has no observed event after the train/test split")
            clients[f"client{index}"] = (x, y)
        if not clients:
            raise ValueError("no client training data remain after splitting")
        return DataSet(
            clients_set=clients,
            train_data=train[feature_names].to_numpy(dtype=np.float32),
            train_label=train[["time", "status"]].to_numpy(dtype=np.float32),
            test_data=test[feature_names].to_numpy(dtype=np.float32),
            test_label=test[["time", "status"]].to_numpy(dtype=np.float32),
            raw_aug_clients_set={},
        )

    def _coerce_features(self, features) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            missing = set(self.feature_names_) - set(features.columns)
            if missing:
                raise ValueError(f"prediction data is missing features: {sorted(missing)}")
            array = features[list(self.feature_names_)].to_numpy(dtype=np.float32)
        else:
            array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.config.n_features:
            raise ValueError(
                f"expected a two-dimensional feature matrix with {self.config.n_features} columns"
            )
        return array

    def _require_fitted(self) -> None:
        if self.runner_ is None:
            raise RuntimeError("fit must be called before prediction or evaluation")
