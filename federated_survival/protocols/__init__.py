"""Federated protocol adapters and the public protocol registry."""

from .base import (
    FederatedProtocol,
    ProtocolCapabilities,
    available_federated_protocols,
    get_federated_protocol,
    register_federated_protocol,
)
from .parameter import FedAvgProtocol, FedOptProtocol, FedProxProtocol
from .webdisco import WebDISCOFitResult, WebDISCOStyleCoxProtocol


for _protocol, _aliases in (
    (FedAvgProtocol, ("fed-avg",)),
    (FedProxProtocol, ("fed-prox",)),
    (FedOptProtocol, ("fedadam", "fed-adam", "fedopt/fedadam")),
    (WebDISCOStyleCoxProtocol, ("webdisco", "webdisco_style", "webdisco-style-cox")),
):
    register_federated_protocol(
        _protocol.name,
        _protocol,
        aliases=_aliases,
    )


__all__ = [
    "FederatedProtocol",
    "ProtocolCapabilities",
    "FedAvgProtocol",
    "FedProxProtocol",
    "FedOptProtocol",
    "WebDISCOFitResult",
    "WebDISCOStyleCoxProtocol",
    "available_federated_protocols",
    "get_federated_protocol",
    "register_federated_protocol",
]
