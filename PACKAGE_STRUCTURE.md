# Package structure

This source tree contains only the software-package implementation,
documentation, examples, and validation tests used by the software article.

## Public entry points

- federated_survival/api.py: high-level FederatedSurvival estimator;
- federated_survival/experiment_api.py: paired Center/Federated/Local
  comparisons, model-protocol matrices, and selectable result plots;
- federated_survival/cli.py: doctor, quickstart, and configuration-driven run
  commands;
- federated_survival/__init__.py: public imports and version metadata.

## Core federated workflow

- federated_survival/core/config.py: validated training configuration;
- federated_survival/core/client.py: local mini-batch optimization;
- federated_survival/core/server.py: server state and aggregation support;
- federated_survival/core/runner.py: end-to-end federated orchestration;
- federated_survival/core/augmenter.py and mvae.py: optional FSA-MVAE
  augmentation;
- federated_survival/core/differential_privacy.py: experimental update clipping
  and perturbation utilities.

## Model and protocol adapters

- federated_survival/models/adapters.py: seven built-in survival-model
  adapters, registry, and survival-prediction validation;
- federated_survival/protocols/base.py: protocol interface and capability
  declarations;
- federated_survival/protocols/parameter.py: FedAvg, FedProx, and FedOpt;
- federated_survival/protocols/webdisco.py: WebDISCO-style distributed linear
  Cox protocol.

## Data and evaluation

- federated_survival/data/generator.py: simulated survival data;
- federated_survival/data/loader.py: tabular data loading;
- federated_survival/data/splitter.py: train-test and client partitioning;
- federated_survival/data/workflow.py: interactive public convenience functions
  for simulation, loading, partitioning, and scatter plots;
- federated_survival/utils/metrics.py: C-index and integrated Brier score.

## Reproducible software checks

- federated_survival/experiments/config_runner.py: JSON/YAML experiment
  execution;
- federated_survival/experiments/infrastructure.py: manifests, environment
  metadata, validation, and artifact hashing;
- federated_survival/examples/: runnable examples and smoke configurations;
- tests/: unit and integration tests;
- .github/workflows/ci.yml: supported-version CI, quickstart, and build checks.

## Documentation and packaging

- docs/ and mkdocs.yml: documentation source;
- pyproject.toml: build metadata, dependencies, entry point, and test settings;
- MANIFEST.in: source-distribution contents;
- LICENSE: MIT License.
