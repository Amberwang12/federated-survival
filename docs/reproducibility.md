# Reproducibility

## Installation diagnostics

~~~bash
federated-survival doctor --output doctor.json
~~~

The report records the package version, Python environment, dependency
versions, and availability of all model and protocol adapters.

## Audited quickstart

~~~bash
federated-survival quickstart \
  --model DeepSurv \
  --protocol FedAvg \
  --seed 42 \
  --output results/quickstart
~~~

The output contains the resolved settings, round-wise metrics, survival
predictions, a summary, and a run manifest with artifact hashes.

## Seven-model software check

~~~bash
federated-survival run \
  --config federated_survival/examples/seven_model_smoke.json \
  --output results/seven_model_smoke
~~~

## Model-protocol coverage check

~~~bash
federated-survival run \
  --config federated_survival/examples/four_protocol_smoke.json \
  --output results/four_protocol_smoke
~~~

Every configuration-driven invocation writes a resolved configuration,
environment metadata, metrics, an explicit failure table, artifact hashes,
timestamps, and final run status. Retain these files with any reported result.

The smoke configurations verify software execution. Their metric values should
not be interpreted as a statistical comparison of models or protocols.

## Function-based comparisons

The composable Python API runs a selected model and federated protocol on
paired train/test and client partitions. For repeatable comparisons, record
the input data version, split seeds, training budgets, package and dependency
versions, and save `result.metrics` alongside generated plots. The function
API does not write a manifest automatically; use the YAML workflow when a
complete audit trail is required. See
[Composable Python experiments](interactive-experiments.md).
