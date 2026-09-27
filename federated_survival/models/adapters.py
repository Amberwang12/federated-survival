"""Unified adapters for survival models supported by the package.

The federated orchestration layer should not need to know whether a model uses
continuous-time Cox targets, a transformed time input, or discretised labels.
Each adapter owns those model-specific operations while exposing one small
contract to clients, servers, and third-party extensions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxCC, CoxPH, CoxTime, DeepHitSingle, LogisticHazard, PCHazard
from pycox.models.cox_time import MLPVanillaCoxTime


class PredictionValidationError(RuntimeError):
    """Raised when a fitted model emits an invalid survival distribution."""


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Machine-readable checks for a survival prediction matrix."""

    n_times: int
    n_samples: int
    finite: bool
    within_probability_bounds: bool
    monotone_non_increasing: bool
    identical_across_samples: bool

    @property
    def valid(self) -> bool:
        return (
            self.n_times >= 2
            and self.n_samples >= 1
            and self.finite
            and self.within_probability_bounds
            and self.monotone_non_increasing
        )


def validate_survival_predictions(
    survival: pd.DataFrame,
    *,
    mode: str = "raise",
    tolerance: float = 1e-6,
) -> PredictionDiagnostics:
    """Validate probability bounds, finiteness, and monotonicity.

    Identical columns are recorded as a degeneracy diagnostic rather than an
    unconditional error because a legitimately uninformative model may emit
    the same curve for every sample.  The experiment layer can nevertheless
    fail a run when that behaviour is unexpected.
    """

    if mode not in {"raise", "warn", "none"}:
        raise ValueError("prediction validation mode must be 'raise', 'warn', or 'none'")
    if not isinstance(survival, pd.DataFrame):
        raise TypeError("survival predictions must be a pandas DataFrame")

    values = survival.to_numpy(dtype=float)
    finite = bool(values.size and np.isfinite(values).all())
    within_bounds = bool(
        finite
        and np.all(values >= -tolerance)
        and np.all(values <= 1.0 + tolerance)
    )
    monotone = bool(
        finite
        and values.shape[0] >= 2
        and np.all(np.diff(values, axis=0) <= tolerance)
    )
    identical = bool(
        values.shape[1] > 1
        and finite
        and np.allclose(values, values[:, [0]], atol=tolerance, rtol=0.0)
    )
    diagnostics = PredictionDiagnostics(
        n_times=int(values.shape[0]) if values.ndim == 2 else 0,
        n_samples=int(values.shape[1]) if values.ndim == 2 else 0,
        finite=finite,
        within_probability_bounds=within_bounds,
        monotone_non_increasing=monotone,
        identical_across_samples=identical,
    )
    if not diagnostics.valid and mode != "none":
        message = (
            "Invalid survival predictions: "
            f"shape={values.shape}, finite={finite}, "
            f"within_probability_bounds={within_bounds}, "
            f"monotone_non_increasing={monotone}. "
            "Inspect the learning rate, privacy-noise scale, and model-specific target settings."
        )
        if mode == "raise":
            raise PredictionValidationError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return diagnostics


#: Largest exponent fed to ``np.exp`` when converting log-cumulative-hazard
#: back to a probability.  ``exp(700)`` is still finite (~1e304) and
#: ``exp(-1e304)`` is exactly 0, so clipping here removes the overflow
#: warnings without changing any representable survival probability.
_LOG_CUMHAZ_CAP = 700.0


def stable_cox_survival(model, features) -> "pd.DataFrame":
    """Cox survival curves ``S(t|x) = exp(-H0(t) * exp(logh))`` computed in log space.

    ``pycox`` evaluates the cumulative hazard as ``H0(t) * exp(logh)``.  When
    the log-hazard is large -- most notably when differential-privacy noise
    swamps the weights, but also for a diverging fit -- ``exp(logh)``
    overflows to ``inf`` and the product collapses to ``inf * 0 = nan`` at
    every time whose baseline cumulative hazard is zero (i.e. all times
    before the first event).  The result is a matrix of ``nan`` values, which
    downstream metrics silently turn into a C-index near chance.

    Adding the logarithms instead keeps the same value while never leaving
    the exponent, so the output is always finite, lies in ``[0, 1]`` and is
    non-increasing in time.
    """
    baseline = model.baseline_cumulative_hazards_
    h0 = np.asarray(baseline, dtype=float).reshape(-1)
    log_hazard = np.asarray(model.predict(np.asarray(features, dtype=np.float32)), dtype=float)
    log_hazard = log_hazard.reshape(-1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # log(0) -> -inf: times before the first event keep S(t|x) = 1, which
        # is the correct value there and, unlike ``inf * 0``, is not a nan.
        log_cumhaz = np.log(h0)[:, None] + log_hazard[None, :]
        log_cumhaz = np.clip(log_cumhaz, -np.inf, _LOG_CUMHAZ_CAP)
        survival = np.exp(-np.exp(log_cumhaz))
    survival = np.clip(survival, 0.0, 1.0)

    index = getattr(baseline, "index", None)
    if index is None:
        index = np.arange(h0.size)
    return pd.DataFrame(survival, index=index)


class _StableCoxSurvivalMixin:
    """Mix in numerically stable survival curves for baseline-hazard Cox models."""

    def predict_survival(self, model, features, *, validation_mode="raise"):
        survival = stable_cox_survival(model, np.asarray(features, dtype=np.float32))
        diagnostics = validate_survival_predictions(survival, mode=validation_mode)
        return survival, diagnostics


class ModelAdapter(ABC):
    """Contract implemented by every built-in and third-party survival model."""
    name: str
    requires_baseline_hazards: bool = False
    uses_label_transform: bool = False

    def create_label_transform(self, config):
        return None

    def configure_targets(self, config, durations, events):
        """Fit any label transform and update output dimensionality."""
        transform = self.create_label_transform(config)
        if transform is None:
            config.labtrans = None
            config.out_features = 1
            return None
        transform.fit_transform(durations, events)
        config.labtrans = transform
        config.out_features = transform.out_features
        return transform

    def transform_target(self, labels: np.ndarray, label_transform=None):
        durations = np.asarray(labels)[:, 0]
        events = np.asarray(labels)[:, 1]
        if label_transform is None:
            return durations, events
        return label_transform.transform(durations, events)

    def build_network(self, config) -> torch.nn.Module:
        return tt.practical.MLPVanilla(
            in_features=config.n_features,
            num_nodes=config.num_nodes,
            out_features=config.out_features,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            activation=config.activation,
            output_bias=False,
        )

    @abstractmethod
    def build_model(self, network, config, label_transform=None, optimizer=None):
        """Wrap a Torch network in the corresponding PyCox model."""

    def prepare_for_prediction(
        self,
        model,
        train_x: np.ndarray,
        train_y: np.ndarray,
        label_transform=None,
    ) -> None:
        """Attach training data and estimate a Cox baseline when required."""
        if not self.requires_baseline_hazards:
            return
        target = self.transform_target(train_y, label_transform)
        model.fit(
            train_x,
            target,
            batch_size=len(train_x),
            epochs=0,
            verbose=False,
        )
        model.compute_baseline_hazards()

    def predict_survival(self, model, features, *, validation_mode="raise"):
        survival = model.predict_surv_df(np.asarray(features, dtype=np.float32))
        diagnostics = validate_survival_predictions(survival, mode=validation_mode)
        return survival, diagnostics


class _CoxPHAdapter(_StableCoxSurvivalMixin, ModelAdapter):
    requires_baseline_hazards = True

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return CoxPH(network, optimizer or torch.optim.Adam)


class CoxPHAdapter(_CoxPHAdapter):
    name = "CoxPH"


class DeepSurvAdapter(_CoxPHAdapter):
    name = "DeepSurv"


class CoxCCAdapter(_StableCoxSurvivalMixin, ModelAdapter):
    name = "CoxCC"
    requires_baseline_hazards = True

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return CoxCC(network, optimizer or torch.optim.Adam)


class CoxTimeAdapter(ModelAdapter):
    name = "CoxTime"
    requires_baseline_hazards = True
    uses_label_transform = True

    def create_label_transform(self, config):
        return CoxTime.label_transform()

    def build_network(self, config) -> torch.nn.Module:
        return MLPVanillaCoxTime(
            in_features=config.n_features,
            num_nodes=config.num_nodes,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            activation=config.activation,
        )

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return CoxTime(
            network,
            optimizer or torch.optim.Adam,
            labtrans=label_transform,
        )


class PCHazardAdapter(ModelAdapter):
    name = "PC-Hazard"
    uses_label_transform = True

    def create_label_transform(self, config):
        return PCHazard.label_transform(config.num_durations, scheme="quantiles")

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return PCHazard(
            network,
            optimizer or torch.optim.Adam,
            duration_index=label_transform.cuts,
        )


class LogisticHazardAdapter(ModelAdapter):
    name = "LogisticHazard"
    uses_label_transform = True

    def create_label_transform(self, config):
        return LogisticHazard.label_transform(config.num_durations, scheme="quantiles")

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return LogisticHazard(
            network,
            optimizer or torch.optim.Adam,
            duration_index=label_transform.cuts,
        )


class DeepHitAdapter(ModelAdapter):
    name = "DeepHit"
    uses_label_transform = True

    def create_label_transform(self, config):
        return DeepHitSingle.label_transform(config.num_durations, scheme="quantiles")

    def build_model(self, network, config, label_transform=None, optimizer=None):
        return DeepHitSingle(
            network,
            optimizer or torch.optim.Adam,
            duration_index=label_transform.cuts,
        )


AdapterType = Union[ModelAdapter, Type[ModelAdapter]]
_REGISTRY: Dict[str, Type[ModelAdapter]] = {}
_CANONICAL_NAMES: Dict[str, str] = {}


def register_model_adapter(
    name: str,
    adapter: Type[ModelAdapter],
    *,
    replace: bool = False,
) -> None:
    """Register a custom adapter under a case-insensitive name."""
    if not isinstance(adapter, type) or not issubclass(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter subclass")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("adapter name cannot be empty")
    if normalized in _REGISTRY and not replace:
        raise ValueError(f"an adapter named {name!r} is already registered")
    _REGISTRY[normalized] = adapter
    _CANONICAL_NAMES[normalized] = name


def get_model_adapter(name: str) -> ModelAdapter:
    """Return a fresh adapter instance for a registered model name."""
    normalized = str(name).strip().lower()
    try:
        adapter = _REGISTRY[normalized]
    except KeyError as error:
        choices = ", ".join(available_model_adapters())
        raise ValueError(f"unsupported model {name!r}; available adapters: {choices}") from error
    return adapter()


def available_model_adapters() -> Tuple[str, ...]:
    """Return canonical built-in and custom names in registration order."""
    return tuple(_CANONICAL_NAMES[name] for name in _REGISTRY)


for _adapter in (
    CoxPHAdapter,
    DeepSurvAdapter,
    CoxCCAdapter,
    CoxTimeAdapter,
    LogisticHazardAdapter,
    PCHazardAdapter,
    DeepHitAdapter,
):
    register_model_adapter(_adapter.name, _adapter)

