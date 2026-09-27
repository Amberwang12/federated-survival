# Federated protocol adapters

Model adapters and protocol adapters solve different extension problems.
`ModelAdapter` contains survival-model-specific target and prediction logic;
`FederatedProtocol` determines what clients exchange and how the server
updates. Version 0.7.0 contains four protocol adapters.

| Protocol | Models | Exchange | Intended use |
| --- | --- | --- | --- |
| FedAvg | all seven adapters | model parameters | default sample-weighted baseline |
| FedProx | all seven adapters | model parameters | reduce local drift under heterogeneous client data |
| FedOpt | all seven adapters | model parameters | adaptive Adam, Yogi, or Adagrad server updates |
| WebDISCO-style | linear CoxPH only | aggregated risk-set statistics | pooled Breslow Cox objective without parameter averaging |

## High-level use

```python
from federated_survival import FederatedSurvival

model = FederatedSurvival(
    model="DeepSurv",
    protocol="FedProx",
    protocol_params={"mu": 0.01},
    n_clients=5,
    global_rounds=20,
    local_steps=2,
).fit(frame)
```

FedOpt defaults to a FedAdam server update:

```python
model = FederatedSurvival(
    model="LogisticHazard",
    protocol="FedOpt",
    protocol_params={
        "optimizer": "adam",  # adam, yogi, or adagrad
        "server_lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 1e-3,
    },
).fit(frame)
```

The survival-specific risk-set-statistics protocol is selected separately:

```python
model = FederatedSurvival(
    model="CoxPH",
    protocol="WebDISCO-style",
    protocol_params={"max_iter": 50, "tolerance": 1e-7},
    n_clients=5,
    client_fraction=1.0,
).fit(frame)
```

`WebDISCO-style` is an independent Python implementation of distributed
Breslow Cox risk-set aggregation. It is not the original WebDISCO web service.
It requires horizontal partitions, full participation, a bias-free linear
CoxPH model, and no client-update perturbation. The reported communication
quantity is an analytical float64 payload estimate and excludes protocol
framing and transport overhead.

## Register a third-party protocol

```python
from federated_survival import FederatedProtocol, register_federated_protocol

class MyProtocol(FederatedProtocol):
    name = "MyProtocol"

    def aggregate(
        self, global_state, client_states, client_weights, *, round_index
    ):
        ...

register_federated_protocol(MyProtocol.name, MyProtocol)
```

Third-party protocols should declare `ProtocolCapabilities`, validate their
supported models and partition geometry, reset persistent server state in
`begin_run`, and report protocol-specific metadata.

## One-command protocol smoke test

```bash
federated-survival run \
  --config federated_survival/examples/four_protocol_smoke.json \
  --output results/four_protocol_smoke
```

This configuration executes 22 cells: seven models under each of FedAvg,
FedProx, and FedOpt, plus CoxPH under WebDISCO-style. Results, failures, the
resolved configuration, and a run manifest are written to the output folder.
