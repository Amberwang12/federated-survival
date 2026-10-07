# Package structure

This source tree contains only the software-package implementation,
documentation, examples, and validation tests used by the software article.

Version 0.8.0. `pyproject.toml`, `federated_survival/__init__.py`, and the
`v0.8.0` git tag all agree; `release.yml` asserts this on every tag push.

## Public entry points

- federated_survival/api.py: high-level FederatedSurvival estimator
  (`n_features` is inferred from the first batch, so callers need not set it);
- federated_survival/experiment_api.py: paired Center/Federated/Local
  comparisons, model-protocol matrices, and selectable result plots;
- federated_survival/cli.py: doctor, quickstart, and configuration-driven run
  commands;
- federated_survival/__init__.py: public imports and version metadata.

## Core federated workflow

- federated_survival/core/config.py: validated training configuration;
- federated_survival/core/client.py: local mini-batch optimization
  (`batch_size` is honoured here, not overridden);
- federated_survival/core/server.py: server state and aggregation support;
- federated_survival/core/runner.py: end-to-end federated orchestration,
  including the `effective_noise_sigma` privacy diagnostic;
- federated_survival/core/augmenter.py and mvae.py: optional FSA-MVAE
  augmentation;
- federated_survival/core/differential_privacy.py: experimental update clipping
  and perturbation utilities. The noise scale includes the 1/sqrt(K) client
  average, and `last_noise_to_signal` reports the realised signal-to-noise
  ratio.

## Model and protocol adapters

- federated_survival/models/adapters.py: seven built-in survival-model
  adapters, registry, and survival-prediction validation. CoxPH-family adapters
  evaluate `S(t|x) = exp(-exp(log H0(t) + logh))` in log space, so a
  `H0(t) = 0` prefix yields `S = 1` instead of the `inf * 0 = NaN` that the
  naive product form produces;
- federated_survival/protocols/base.py: protocol interface and capability
  declarations;
- federated_survival/protocols/parameter.py: FedAvg, FedProx, and FedOpt;
- federated_survival/protocols/webdisco.py: WebDISCO-style distributed linear
  Cox protocol.

Model preparation (output width, label transform, network construction) is
owned by the adapter registry. `experiments/baselines.py` delegates to it
rather than reimplementing it, and `tests/test_baseline_adapter_parity.py`
pins the numerical fingerprint of all seven built-in models so the Center and
Local baselines cannot silently drift from the federated path.

## Data and evaluation

- federated_survival/data/generator.py: simulated survival data;
- federated_survival/data/loader.py: tabular data loading;
- federated_survival/data/splitter.py: train-test and client partitioning.
  `iid`, `non-iid`, `time-non-iid`, `dirichlet`, and `random` are dispatched
  case-insensitively;
- federated_survival/data/workflow.py: interactive public convenience functions
  for simulation, loading, partitioning, and scatter plots, plus
  `load_real_data()` and `available_real_datasets()`;
- federated_survival/data/real/: bundled GBSG breast-cancer and colon-cancer
  tables, shipped inside the wheel and sdist via the `data/real/*.csv`
  package-data rule so `pip install` users can load them without cloning the
  repository;
- federated_survival/utils/metrics.py: C-index and integrated Brier score.

The repository-root `data/real/` copies exist for documentation and
experiments; `MANIFEST.in` excludes them from the source distribution on
purpose, since the package-internal copies are the shipped ones.

## Reproducible software checks

- federated_survival/experiments/config_runner.py: JSON/YAML experiment
  execution;
- federated_survival/experiments/config_schema.py: configuration validation.
  `references.optimizer_steps` defaults to `global_rounds * local_steps` so
  reference baselines are budget-matched unless you override it explicitly;
- federated_survival/experiments/infrastructure.py: manifests, environment
  metadata, validation, and artifact hashing;
- federated_survival/experiments/baselines.py: Center and Local reference
  training;
- federated_survival/experiments/visualization.py: result plots and the
  `mean_ci95` error band;
- federated_survival/experiments/workflow_runner.py: end-to-end experiment
  workflows;
- federated_survival/examples/: runnable examples and smoke configurations.
  This directory intentionally has no `__init__.py`, so it is not a subpackage
  and is excluded from coverage reporting;
- tests/: unit and integration tests (29 `test_*.py` files, 288 cases);
- experiments/ at the repository root: reviewer-reproduction and audit
  scripts, included in the sdist;
- examples/ at the repository root: the paper-style numbered scripts
  (01-11) and `examples/compare/`;
- scripts/: example batch runners and the coverage gap analyzer;
- .github/workflows/ci.yml: supported-version CI (Python 3.8-3.14), an 80%
  coverage floor, the `federated-survival doctor` smoke check, and build plus
  `twine check --strict`;
- .github/workflows/release.yml: tag-triggered build, tag/version consistency
  assertion, checksums, and GitHub Release publication;
- .github/workflows/docs.yml: MkDocs build and GitHub Pages deployment.

## Documentation and packaging

- docs/ and mkdocs.yml: documentation source;
- pyproject.toml: build metadata, dependencies, entry point, and test settings.
  `license = { text = "MIT" }` is used instead of the bare PEP 639 string form
  because the latter requires setuptools >= 77, which is unavailable on
  Python 3.8;
- MANIFEST.in: source-distribution contents;
- CHANGELOG.md: release history in Keep a Changelog format;
- LICENSE: MIT License;
- PACKAGE_STRUCTURE.md: this file, shipped in the sdist.
