"""Horizontal parameter-exchange protocols."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch

from .base import FederatedProtocol, ProtocolCapabilities


def weighted_average_state(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    client_weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """Return a normalized sample-weighted state-dict average."""
    if not client_states or len(client_states) != len(client_weights):
        raise ValueError("client_states and client_weights must have the same non-zero length")
    weights = torch.as_tensor(client_weights, dtype=torch.float64)
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("client weights must be finite with a positive sum")
    weights = weights / weights.sum()
    keys = tuple(client_states[0].keys())
    if any(tuple(state.keys()) != keys for state in client_states[1:]):
        raise ValueError("all client state dictionaries must have identical keys")

    averaged: Dict[str, torch.Tensor] = {}
    for key in keys:
        first = client_states[0][key]
        if first.is_floating_point() or first.is_complex():
            value = torch.zeros_like(first)
            for coefficient, state in zip(weights, client_states):
                value.add_(state[key].to(value.device, value.dtype), alpha=float(coefficient))
            averaged[key] = value
        else:
            # Integer counters (for example BatchNorm num_batches_tracked)
            # are not parameters and have no meaningful weighted average.
            averaged[key] = first.detach().clone()
    return averaged


class ProximalLoss(torch.nn.Module):
    """Add the FedProx distance-to-broadcast-model penalty to a pycox loss."""

    def __init__(
        self,
        base_loss: torch.nn.Module,
        named_parameters,
        global_state: Mapping[str, torch.Tensor],
        mu: float,
    ) -> None:
        super().__init__()
        if mu < 0:
            raise ValueError("FedProx mu must be non-negative")
        self.base_loss = base_loss
        self.mu = float(mu)
        self._parameters_and_anchors = []
        for name, parameter in named_parameters:
            if name not in global_state:
                raise ValueError(f"global state is missing parameter {name!r}")
            anchor = global_state[name].detach().clone()
            self._parameters_and_anchors.append((parameter, anchor))

    def forward(self, *args, **kwargs):
        value = self.base_loss(*args, **kwargs)
        if self.mu == 0:
            return value
        penalty = value.new_zeros(())
        for parameter, anchor in self._parameters_and_anchors:
            reference = anchor.to(device=parameter.device, dtype=parameter.dtype)
            penalty = penalty + torch.sum((parameter - reference) ** 2)
        return value + 0.5 * self.mu * penalty


class FedAvgProtocol(FederatedProtocol):
    name = "FedAvg"
    capabilities = ProtocolCapabilities()

    def aggregate(
        self,
        global_state,
        client_states,
        client_weights,
        *,
        round_index,
    ):
        return weighted_average_state(client_states, client_weights)


class FedProxProtocol(FedAvgProtocol):
    name = "FedProx"

    def prepare_client_model(self, model, global_state) -> None:
        model.loss = ProximalLoss(
            model.loss,
            model.net.named_parameters(),
            global_state,
            self.config.proximal_mu,
        )

    def metadata(self):
        result = super().metadata()
        result["proximal_mu"] = float(self.config.proximal_mu)
        return result


class FedOptProtocol(FederatedProtocol):
    """FedOpt with Adam, Yogi, or Adagrad server-side adaptation."""

    name = "FedOpt"
    capabilities = ProtocolCapabilities()

    def begin_run(self, config, global_state) -> None:
        super().begin_run(config, global_state)
        self._step = 0
        self._first_moment: Dict[str, torch.Tensor] = {}
        self._second_moment: Dict[str, torch.Tensor] = {}

    def aggregate(
        self,
        global_state,
        client_states,
        client_weights,
        *,
        round_index,
    ):
        average = weighted_average_state(client_states, client_weights)
        self._step += 1
        beta1 = float(self.config.server_beta1)
        beta2 = float(self.config.server_beta2)
        learning_rate = float(self.config.server_learning_rate)
        tau = float(self.config.server_tau)
        algorithm = self.config.server_optimizer
        result: Dict[str, torch.Tensor] = {}

        for name, global_value in global_state.items():
            if not (global_value.is_floating_point() or global_value.is_complex()):
                result[name] = average[name].detach().clone()
                continue
            # A weighted client delta is treated as a server pseudo-gradient.
            gradient = global_value.detach() - average[name].to(
                device=global_value.device, dtype=global_value.dtype
            )
            first = self._first_moment.setdefault(name, torch.zeros_like(global_value))
            second = self._second_moment.setdefault(name, torch.zeros_like(global_value))
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)

            if algorithm == "adam":
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            elif algorithm == "yogi":
                squared = gradient * gradient
                second.addcmul_(
                    torch.sign(squared - second),
                    squared,
                    value=-(1.0 - beta2),
                )
                second.clamp_(min=0)
            elif algorithm == "adagrad":
                second.addcmul_(gradient, gradient, value=1.0)
            else:  # guarded by FSAConfig, retained for third-party configs
                raise ValueError(f"unsupported FedOpt server optimizer {algorithm!r}")

            if algorithm in {"adam", "yogi"}:
                first_hat = first / (1.0 - beta1 ** self._step)
                second_hat = second / (1.0 - beta2 ** self._step)
            else:
                first_hat = first / (1.0 - beta1 ** self._step)
                second_hat = second
            result[name] = global_value.detach() - learning_rate * first_hat / (
                torch.sqrt(second_hat) + tau
            )
        return result

    def metadata(self):
        result = super().metadata()
        result.update(
            {
                "server_optimizer": self.config.server_optimizer,
                "server_learning_rate": float(self.config.server_learning_rate),
                "server_beta1": float(self.config.server_beta1),
                "server_beta2": float(self.config.server_beta2),
                "server_tau": float(self.config.server_tau),
            }
        )
        return result

