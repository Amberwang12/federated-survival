"""Public protocol contract and registry for federated training.

Model adapters answer *how a survival model is constructed and predicts*.
Protocol adapters answer *what clients exchange and how the server updates*.
Keeping those responsibilities separate allows one model adapter to be reused
with several horizontal parameter-aggregation protocols while also permitting
survival-specific protocols such as distributed Cox risk-set aggregation.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Type

import torch


@dataclass(frozen=True)
class ProtocolCapabilities:
    """Machine-readable compatibility declaration for a protocol."""

    partition_geometry: str = "horizontal"
    supported_models: Optional[Tuple[str, ...]] = None
    supports_minibatch: bool = True
    supports_partial_participation: bool = True
    exchanges_model_parameters: bool = True
    requires_risk_set_statistics: bool = False


class FederatedProtocol(ABC):
    """Base class for built-in and third-party federated protocols."""

    name: str
    capabilities = ProtocolCapabilities()
    owns_training_loop: bool = False

    def __init__(self) -> None:
        self.config = None

    def validate_config(self, config) -> None:
        supported = self.capabilities.supported_models
        if supported is not None and config.model_type not in supported:
            choices = ", ".join(supported)
            raise ValueError(
                f"protocol {self.name!r} supports only these models: {choices}"
            )
        if (
            not self.capabilities.supports_partial_participation
            and config.client_sample_ratio != 1.0
        ):
            raise ValueError(f"protocol {self.name!r} requires full client participation")
        if not self.capabilities.supports_minibatch and not config.full_batch:
            # A protocol-owned loop may consume complete local risk-set
            # summaries without using the generic ``full_batch`` switch.
            if not self.owns_training_loop:
                raise ValueError(f"protocol {self.name!r} does not support mini-batches")

    def begin_run(self, config, global_state: Mapping[str, torch.Tensor]) -> None:
        """Reset protocol state before a new fit."""
        self.config = config

    def prepare_client_model(self, model, global_state: Mapping[str, torch.Tensor]) -> None:
        """Optionally modify the local objective before optimizer steps."""

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        client_states: Sequence[Mapping[str, torch.Tensor]],
        client_weights: Sequence[float],
        *,
        round_index: int,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def round_communication_bytes(self, n_clients: int, model_bytes: int) -> int:
        """Return the payload-byte estimate for one parameter-exchange round."""
        return int(2 * n_clients * model_bytes)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "capabilities": asdict(self.capabilities),
        }


ProtocolType = Type[FederatedProtocol]
_REGISTRY: Dict[str, ProtocolType] = {}
_CANONICAL_NAMES: Dict[str, str] = {}
_ALIASES: Dict[str, str] = {}


def register_federated_protocol(
    name: str,
    protocol: ProtocolType,
    *,
    aliases: Sequence[str] = (),
    replace: bool = False,
) -> None:
    """Register a protocol class under a case-insensitive public name."""
    if not isinstance(protocol, type) or not issubclass(protocol, FederatedProtocol):
        raise TypeError("protocol must be a FederatedProtocol subclass")
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("protocol name cannot be empty")
    if normalized in _REGISTRY and not replace:
        raise ValueError(f"a protocol named {name!r} is already registered")
    _REGISTRY[normalized] = protocol
    _CANONICAL_NAMES[normalized] = name
    for alias in aliases:
        alias_key = str(alias).strip().lower()
        if not alias_key:
            raise ValueError("protocol alias cannot be empty")
        if alias_key in _ALIASES and not replace:
            raise ValueError(f"a protocol alias named {alias!r} is already registered")
        _ALIASES[alias_key] = normalized


def get_federated_protocol(name: str) -> FederatedProtocol:
    """Return a fresh protocol instance from the public registry."""
    normalized = str(name).strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    try:
        protocol = _REGISTRY[normalized]
    except KeyError as error:
        choices = ", ".join(available_federated_protocols())
        raise ValueError(
            f"unsupported federated protocol {name!r}; available protocols: {choices}"
        ) from error
    return protocol()


def available_federated_protocols() -> Tuple[str, ...]:
    """Return canonical protocol names in registration order."""
    return tuple(_CANONICAL_NAMES[name] for name in _REGISTRY)

