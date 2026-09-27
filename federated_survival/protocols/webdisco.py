"""A WebDISCO-style horizontally distributed linear Cox protocol.

This implementation follows the defining statistical idea of WebDISCO:
clients exchange aggregated risk-set statistics and the server performs a
Newton update for the pooled Breslow Cox partial likelihood.  It is an
independent Python implementation, not the original WebDISCO web service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .base import FederatedProtocol, ProtocolCapabilities


@dataclass(frozen=True)
class WebDISCOFitResult:
    coefficients: np.ndarray
    coefficient_history: Tuple[np.ndarray, ...]
    log_likelihood_history: Tuple[float, ...]
    score_norm: float
    converged: bool
    n_iterations: int
    communication_bytes: int


class WebDISCOStyleCoxProtocol(FederatedProtocol):
    """Distributed Newton solver for a linear pooled CoxPH objective."""

    name = "WebDISCO-style"
    owns_training_loop = True
    capabilities = ProtocolCapabilities(
        partition_geometry="horizontal",
        supported_models=("CoxPH",),
        supports_minibatch=False,
        supports_partial_participation=False,
        exchanges_model_parameters=False,
        requires_risk_set_statistics=True,
    )

    def validate_config(self, config) -> None:
        super().validate_config(config)
        if config.use_differential_privacy:
            raise ValueError(
                "WebDISCO-style does not use the client-update perturbation module"
            )

    @staticmethod
    def _coerce_clients(client_sets: Mapping) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        result = {}
        feature_count = None
        for client_id, (features, labels) in client_sets.items():
            x = np.asarray(features, dtype=np.float64)
            y = np.asarray(labels, dtype=np.float64)
            if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 2 or len(x) != len(y):
                raise ValueError("each WebDISCO client must contain X and [time, event] arrays")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("WebDISCO-style inputs must be finite")
            if not np.isin(y[:, 1], [0.0, 1.0]).all():
                raise ValueError("event indicators must be binary")
            if feature_count is None:
                feature_count = x.shape[1]
            elif x.shape[1] != feature_count:
                raise ValueError("all WebDISCO clients must have the same feature dimension")
            result[str(client_id)] = (x, y)
        if not result:
            raise ValueError("WebDISCO-style requires at least one client")
        if sum(int(y[:, 1].sum()) for _, y in result.values()) == 0:
            raise ValueError("WebDISCO-style requires at least one observed event")
        return result

    @staticmethod
    def event_times(client_sets: Mapping) -> np.ndarray:
        clients = WebDISCOStyleCoxProtocol._coerce_clients(client_sets)
        values = [y[y[:, 1] > 0, 0] for _, y in clients.values()]
        return np.unique(np.concatenate(values))

    @staticmethod
    def _risk_maxima(x, y, beta, event_times):
        linear_predictor = x @ beta
        maxima = np.full(len(event_times), -np.inf, dtype=np.float64)
        durations = y[:, 0]
        for index, time in enumerate(event_times):
            risk = durations >= time
            if risk.any():
                maxima[index] = float(np.max(linear_predictor[risk]))
        return maxima

    @staticmethod
    def _local_statistics(x, y, beta, event_times, global_maxima):
        n_times = len(event_times)
        n_features = x.shape[1]
        s0 = np.zeros(n_times, dtype=np.float64)
        s1 = np.zeros((n_times, n_features), dtype=np.float64)
        s2 = np.zeros((n_times, n_features, n_features), dtype=np.float64)
        event_count = np.zeros(n_times, dtype=np.float64)
        event_x_sum = np.zeros((n_times, n_features), dtype=np.float64)
        event_eta_sum = np.zeros(n_times, dtype=np.float64)
        durations, events = y[:, 0], y[:, 1]
        eta = x @ beta
        for index, time in enumerate(event_times):
            risk = durations >= time
            event = (durations == time) & (events > 0)
            shifted = np.exp(eta[risk] - global_maxima[index])
            risk_x = x[risk]
            s0[index] = shifted.sum()
            s1[index] = np.sum(risk_x * shifted[:, None], axis=0)
            s2[index] = np.einsum("ni,nj,n->ij", risk_x, risk_x, shifted)
            event_count[index] = event.sum()
            if event.any():
                event_x_sum[index] = x[event].sum(axis=0)
                event_eta_sum[index] = eta[event].sum()
        return s0, s1, s2, event_count, event_x_sum, event_eta_sum

    def distributed_objective(self, client_sets, beta, event_times=None):
        """Return pooled Breslow log likelihood, score, information and bytes."""
        clients = self._coerce_clients(client_sets)
        beta = np.asarray(beta, dtype=np.float64).reshape(-1)
        if event_times is None:
            event_times = self.event_times(clients)
        event_times = np.asarray(event_times, dtype=np.float64)
        local_maxima = np.stack(
            [self._risk_maxima(x, y, beta, event_times) for x, y in clients.values()]
        )
        global_maxima = np.max(local_maxima, axis=0)
        if not np.isfinite(global_maxima).all():
            raise FloatingPointError("an event time has an empty global risk set")

        totals = None
        for x, y in clients.values():
            values = self._local_statistics(x, y, beta, event_times, global_maxima)
            if totals is None:
                totals = [value.copy() for value in values]
            else:
                for total, value in zip(totals, values):
                    total += value
        s0, s1, s2, event_count, event_x_sum, event_eta_sum = totals
        if np.any(s0 <= 0):
            raise FloatingPointError("non-positive distributed Cox risk-set denominator")

        means = s1 / s0[:, None]
        score = np.sum(event_x_sum - event_count[:, None] * means, axis=0)
        information = np.zeros((len(beta), len(beta)), dtype=np.float64)
        for index in range(len(event_times)):
            covariance = s2[index] / s0[index] - np.outer(means[index], means[index])
            information += event_count[index] * covariance
        log_likelihood = float(
            np.sum(
                event_eta_sum
                - event_count * (global_maxima + np.log(s0))
            )
        )

        l2 = float(self.config.webdisco_l2) if self.config is not None else 0.0
        if l2:
            log_likelihood -= 0.5 * l2 * float(beta @ beta)
            score -= l2 * beta
            information += l2 * np.eye(len(beta))

        n_clients = len(clients)
        n_times = len(event_times)
        n_features = len(beta)
        # Analytical float64 payload count. It excludes serialization and
        # protocol framing and is therefore labelled as a payload estimate.
        first_pass = 2 * n_clients * n_times
        second_pass = n_clients * n_times * (
            2 + 2 * n_features + n_features * n_features
        )
        coefficient_broadcast = n_clients * n_features
        payload_bytes = int(8 * (first_pass + second_pass + coefficient_broadcast))
        return log_likelihood, score, information, payload_bytes

    def fit_coefficients(self, client_sets) -> WebDISCOFitResult:
        clients = self._coerce_clients(client_sets)
        n_features = next(iter(clients.values()))[0].shape[1]
        event_times = self.event_times(clients)
        beta = np.zeros(n_features, dtype=np.float64)
        beta_history = []
        likelihood_history = []
        total_bytes = 0
        converged = False
        final_score_norm = float("inf")

        for _ in range(int(self.config.webdisco_max_iter)):
            likelihood, score, information, payload = self.distributed_objective(
                clients, beta, event_times
            )
            total_bytes += payload
            final_score_norm = float(np.linalg.norm(score))
            ridge = float(self.config.webdisco_ridge)
            try:
                step = np.linalg.solve(
                    information + ridge * np.eye(n_features), score
                )
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(
                    information + ridge * np.eye(n_features)
                ) @ score

            step_norm = float(np.linalg.norm(step))
            max_step = float(self.config.webdisco_max_step_norm)
            if step_norm > max_step:
                step *= max_step / step_norm

            scale = 1.0
            accepted = False
            candidate_likelihood = likelihood
            while scale >= 2.0 ** -12:
                candidate = beta + scale * step
                candidate_likelihood, _, _, candidate_payload = self.distributed_objective(
                    clients, candidate, event_times
                )
                total_bytes += candidate_payload
                if candidate_likelihood >= likelihood - 1e-12:
                    beta = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break

            beta_history.append(beta.copy())
            likelihood_history.append(float(candidate_likelihood))
            if scale * step_norm <= float(self.config.webdisco_tolerance):
                converged = True
                break

        final_likelihood, final_score, _, payload = self.distributed_objective(
            clients, beta, event_times
        )
        total_bytes += payload
        final_score_norm = float(np.linalg.norm(final_score))
        if final_score_norm <= max(float(self.config.webdisco_tolerance), 1e-6):
            converged = True
        if not beta_history:
            beta_history.append(beta.copy())
            likelihood_history.append(float(final_likelihood))
        return WebDISCOFitResult(
            coefficients=beta.copy(),
            coefficient_history=tuple(beta_history),
            log_likelihood_history=tuple(likelihood_history),
            score_norm=final_score_norm,
            converged=converged,
            n_iterations=len(beta_history),
            communication_bytes=total_bytes,
        )

    def aggregate(self, *args, **kwargs):
        raise RuntimeError("WebDISCO-style owns its risk-set-statistics training loop")

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "implementation_scope": (
                    "independent Breslow linear-Cox risk-set-statistics implementation; "
                    "not the original WebDISCO web service"
                ),
                "max_iterations": int(self.config.webdisco_max_iter),
                "tolerance": float(self.config.webdisco_tolerance),
                "l2": float(self.config.webdisco_l2),
            }
        )
        return result

