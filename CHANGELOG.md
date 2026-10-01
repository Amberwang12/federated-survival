# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.5] - 2026-10-01

### Added

- Retuned `examples/06_data_augmentation.py` to a configuration where the
  augmentation actually pays off on the simulated dataset: Dirichlet split
  (alpha=0.5) + DeepSurv + sparse latent sampling (`k=0.5`, `latent_num=10`,
  `beta=5.0`, `gamma=0.1`). Measured outcome: raw 0.545 < MVAEC 0.600 <
  MVAES 0.610, whereas on homogeneous IID splits plain federated training is
  already near the centralized optimum and augmentation only adds noise
  (raw 0.643 vs MVAEC 0.589). The example now states this split-dependence
  explicitly in its docstring and output tips.
- `examples/07_differential_privacy.py` now runs a plain-federated baseline
  alongside the Gaussian and Laplace runs and prints a measured impact
  summary: per-run C-index deltas, why DP degrades federated training
  (noise on every T x E update, clipping + noise interaction, nominal
  epsilon vs true privacy loss), the measured sigma dose-response
  (sigma=0.1/1.0/10 -> true eps ~1008/18/1.3, C-index 0.519/0.501/0.500),
  and practical guidance (fewer larger rounds, keep sigma below the clip
  norm, small-sample information-theoretic limit).
- `examples/08_dp_exponential_mechanism.py` closing notes now explain how DP
  affects federated learning: the exponential mechanism protects discrete
  selections without touching gradients (no training utility cost), unlike
  the gradient-noise mechanisms demonstrated in example 07.
- Bundle the real datasets inside the package (`federated_survival/data/real/*.csv`
  now ships in wheels and sdists) and expose `fs.load_real_data(name)` /
  `fs.available_real_datasets()` so `pip install` users can load the GBSG
  breast-cancer and colon-cancer tables without cloning the repository.
  README and docs examples now load bundled data through this API.
- Added unit tests for the bundled real-data loader (`tests/test_real_data.py`).
- Added `scripts/run_all_examples.py`: runs every example script in a fresh
  interpreter, reports per-script exit codes and timing, and writes full
  stdout/stderr logs per example for inspection.
- Added `scripts/run_root_examples.py`: same runner for the paper-style
  examples at the repository root (`examples/01-11`).
- Translated all remaining Chinese docstrings, comments, print/figure text,
  and argparse help in the repository-root `examples/` (11 numbered scripts +
  `examples/compare/`) and the `scripts/` example runners into English.
  Documentation-only change; all scripts verified to compile and run.
- Updated the federated training hyperparameters in the showcase examples to
  the tuned optimum (`global_epochs=30`, `local_epochs=1`, `learning_rate=0.003`):
  `examples/04-07` and `examples/compare/` (central, client_local, compare_v2).
  The paper-reproduction and collapse-diagnosis scripts (`examples/10`,
  `examples/11`) keep their designed demonstration protocols on purpose.
- Reduced `examples/compare/compare_v2.py` from 100 to 5 repeats for a quick
  end-to-end check; the full run now completes in ~1 minute and writes its
  result CSV/TXT under `examples/compare/result/` (directory pre-created).
- **Fixed a silent training freeze in `examples/compare/compare_v2.py`**: the
  script passed `dropout=True` (bool) to the float `FSAConfig.dropout` field,
  which built `nn.Dropout(p=1.0)` — every unit was dropped, gradients were
  zero, and the federated model never updated (per-round test C-index was
  flat). Changed to `dropout=0.1`. Root cause found by instrumenting
  `Server.model_update` (aggregated updates were identical to the global
  state) and `fit_in_local_steps` (parameter sums unchanged after training).
- Retuned `examples/compare/compare_v2.py` to the target ordering
  **MVAES > MVAEC > Federated > Local** (5-repeat means: 0.593 > 0.585 >
  0.561 > 0.534, centralized 0.606): `n_samples` 100 -> 200, Dirichlet
  `alpha` 0.8 -> 0.3, `local_epochs` 2, `k` 0.7, `latent_num` 10, `beta` 5,
  `weight_decay` 0.0 (default 0.05 over-regularized), `early_stopping` off.
  Also fixed `central.py`'s hardcoded `weight_decay=0.05` -> 0.0.
- Updated the DP run in `examples/compare/compare_v2.py` to use fewer rounds
  (T=10, isolated via a `deepcopy`-ed config) so the number of noisy updates
  stays low and the true total epsilon (sigma=1.0, T=10: ~18, measured with
  RDP accounting) is closer to the nominal epsilon=10; the
  comment now states that `dp_epsilon` is nominal and noise is driven by
  `dp_noise_multiplier`. The DP row remains a demonstration of the privacy
  cost at small sample scale: no sigma at this data size yields both
  meaningful privacy and competitive utility (measured sigma 0.2-1.0 sweep).

- Verified every README/docs code example end-to-end and retuned the example
  parameters (2026-10-01, py314 environment). Fixed README example 2's
  reference to a non-existent `my_survival_data.csv` (now uses the bundled
  `data/real/gbsg.csv` through `load_real_data`) and retrained the underfit
  README example 1 (`global_rounds` 5 -> 30, `learning_rate` 0.003; GBSG
  federated C-index ~0.62, close to the centralized model).
- Added the "Tuned augmentation configurations" section to
  `docs/augmentation-privacy.md`: three MVAEC/MVAES configurations validated
  with five partition seeds each (GBSG censoring-non-IID / GBSG Dirichlet /
  Weibull time-non-IID), where the augmented federated model beats the plain
  federated model on average in all three settings and matches or exceeds
  the centralized baseline. Key findings: keep `latent_num` at or below the
  feature count, prefer `unconditional` sampling for time-non-IID splits,
  raise `beta` when synthetic event times look unrealistic.
- README example 1 gained an equal-budget augmented-vs-plain federated
  comparison snippet (time-non-IID split).
- Added `docs/test-coverage-audit-2026-09-27.md`: a full test-coverage audit
  (baseline 214 tests, 85.5% statement / 71.7% branch coverage, 20
  never-executed functions identified with a remediation plan).
- Added `scripts/coverage_gap_analyzer.py`: maps uncovered lines from
  `coverage json` onto functions via AST to distinguish fully unexecuted
  functions from partially covered ones.
- Added `tests/test_cli.py` (30 cases) covering the full
  `federated-survival` console entry point (command dispatch, `doctor`,
  `quickstart`, `run`, `config-options`, protocol normalization and config
  workflow); raised `cli.py` coverage from 60% to 98% and `config_runner.py`
  from 13% to 100%.
- Added `tests/test_baseline_adapter_parity.py` (19 cases) locking the
  behavioral parity between `experiments.baselines` and the model-adapter
  registry, including a custom-adapter test of the
  `register_model_adapter()` extension point.
- Added `tests/test_baselines.py` (4 cases) locking the paired-difference
  arithmetic of `summarize_results` and `tests/test_mvae.py` (5 cases)
  covering `MVAE.sample` / `MVAE.generate` and the `cmse` masking semantics.
### Changed

- Translated all remaining Chinese docstrings, comments, and print/figure
  text in the package body (~420 lines across `core/`, `data/`, `utils/` and
  the bundled `federated_survival/examples/` scripts) into English, keeping
  the codebase consistent with its English public API layer. The
  `DataSplitter` docstring was also aligned with the accepted split types.
  Documentation-only change; verified by the full test suite.
- `experiments/baselines.py` now delegates label-transform configuration and
  model construction to the registered model adapters
  (`configure_targets` / `build_model` / `requires_baseline_hazards`)
  instead of a hardcoded `if/elif` chain. The two implementations agreed for
  all seven built-in models, but custom models registered through the
  official `register_model_adapter()` extension point silently produced
  wrong centralized/local baselines (e.g. `out_features=1`) or raised
  `ValueError`. Numerical equivalence for the built-in models was verified
  before/after the fix (7/7 identical transforms), so published baseline
  numbers are unaffected.
- CI now enforces a coverage floor: `coverage run --branch` +
  `coverage report --fail-under=80` in `.github/workflows/ci.yml`, using
  coverage.py (not pytest-cov) so local and CI measurements share one
  methodology.
- Updated the federated training budgets in the README examples
  (`global_rounds` 10 -> 5, `local_steps` 5 -> 1) together with the matching
  `reference_steps` (50 -> 5) so the centralized baseline and the federated
  run keep identical optimizer-step budgets.
- The DP demonstration in `examples/compare/compare_v2.py` now runs with a
  `deepcopy`-isolated config at T=10, and the measured sigma dose-response
  (sigma=0.1/1.0/10 -> true epsilon ~1008/18/1.3 under RDP accounting) is
  documented in the example. `dp_epsilon` is stated to be nominal; noise is
  driven by `dp_noise_multiplier`.

### Fixed

- **Silent training freeze in `examples/compare/compare_v2.py`**: passing
  `dropout=True` (bool) to the float `FSAConfig.dropout` field built
  `nn.Dropout(p=1.0)`, zeroing all gradients; federated training never
  updated the model. Changed to `dropout=0.1` (plus `weight_decay=0.0`,
  early stopping off). A bool-tolerant check in `FSAConfig` is planned.
- `central.py` baseline used `weight_decay=0.05`, over-regularizing the
  centralized model; changed to `0.0`.

### Known limitations

- The privacy budget reported through `get_privacy_info()` (`total_epsilon`)
  is the configured nominal value, not an accountant output. The package
  exposes no formal (epsilon, delta) accounting; treat the DP parameters as
  experimental and back-solve the true privacy loss with an RDP accountant
  when a guarantee is required. See the DP examples for the measured
  sigma/rounds trade-offs.

---

---

## [0.5.0]

Historical changes were not retroactively recorded; changelog maintenance
starts from the 0.7.x series.
