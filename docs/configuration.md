# Configuration-driven experiments

The schema-v1 experiment format turns one YAML file into a complete workflow:

1. load and validate real or simulated survival data;
2. create one held-out test set and federated client partitions;
3. run centralized, federated, and isolated-local references on paired seeds;
4. export final metrics, round metrics, predictions, diagnostics, and an audit manifest;
5. generate publication or presentation figures without hand-written plotting code.

For a data-generation and partition preview without model training, use:

```bash
federated-survival run \
  --config federated_survival/examples/simulation_partition_scatter.yaml
```

This writes the generated observations, train/client partition tables, audit
files, and a client-coloured scatter plot. No Python driver script is needed.

For experiments assembled interactively without writing a configuration,
see [Composable Python experiments](interactive-experiments.md).

Input file paths are resolved relative to the YAML file. Output directories are
resolved relative to the shell working directory. Existing legacy smoke YAML
and JSON files remain supported.

## `experiment`

```yaml
experiment:
  stage: full
  seeds: [0, 1, 2, 3, 4]
  fail_fast: false
```

- `stage: data_only` stops after data generation/loading, train/test splitting,
  client partitioning, CSV export, and partition visualization.
- `stage: full` continues through model training and evaluation.

- `seeds` controls the train/test split, client partition, and model
  initialization. Center, federated, and local methods therefore form paired
  comparisons within every seed.
- Use one seed for a connectivity check, at least five for an exploratory
  distribution, and a prespecified larger number for a formal analysis.
- Boxplots and violin plots require at least two seeds. The configuration is
  rejected instead of drawing a degenerate one-observation box.
- `fail_fast: false` records a traceback and continues with later seeds.

## `data`

Real CSV input:

```yaml
data:
  source: csv
  label: cohort
  path: ../data/real/cohort.csv
  duration_column: time
  event_column: status
  test_size: 0.2
  standardize: true
  missing_values: error
```

Excel uses `source: excel`. Simulation uses:

```yaml
data:
  source: simulation
  mechanism: weibull
  n_samples: 1000
  n_features: 20
  censoring_rate: 0.4
  test_size: 0.2
```

Feature columns must be numeric. Observed times must be positive and finite;
event values must be zero or one. Missing-value choices are:

`label` is an optional human-readable dataset name used in composite figures.

- `error`: stop before training; recommended for audited experiments;
- `drop`: remove rows containing missing features;
- `median`: use medians calculated from the training set only, then apply them
  to clients and the test set.

When `standardize: true`, means and standard deviations are also estimated
from pooled training data only. Test observations never determine a
preprocessing parameter.

For `client_partition.type: scatter`, axes can be `time` or `feature_N`, where
`feature_1` is the first input feature after preprocessing. For example:

```yaml
visualization:
  enabled: true
  formats: [png, pdf]
  client_partition:
    type: scatter
    x: feature_1
    y: time
```

## `partition`

```yaml
partition:
  method: iid
  n_clients: 3
  stratify_by: status
  alpha: 0.5
```

Available methods are:

| Value | Meaning | Typical use |
|---|---|---|
| `iid` | event-stratified equal client split | baseline comparisons |
| `random` | unstratified random client split | IID sensitivity |
| `censoring-non-iid` | client censoring-rate shift | event-rate heterogeneity |
| `time-non-iid` | client observed-time shift | follow-up heterogeneity |
| `dirichlet` | joint time/event Dirichlet allocation | general heterogeneity |

`alpha` affects only `dirichlet`; smaller values create stronger
heterogeneity. Schema version 1 uses an event-stratified train/test split, so
`stratify_by` must be `status`.

## `model`

```yaml
model:
  name: DeepSurv
  hidden_nodes: [32, 32]
  activation: relu
  dropout: 0.1
  weight_decay: 0.0
  num_durations: 25
```

Model choices are `CoxPH`, `DeepSurv`, `CoxCC`, `CoxTime`,
`LogisticHazard`, `PC-Hazard`, and `DeepHit`. Activation choices are `relu`,
`tanh`, `softplus`, and `sigmoid`. CoxPH requires `hidden_nodes: []`.
`num_durations` is primarily relevant to discrete-time models but remains in
the resolved configuration for complete provenance.

## `federated`

```yaml
federated:
  protocol: FedAvg
  global_rounds: 10
  local_steps: 5
  client_fraction: 1.0
  batch_mode: mini_batch
  batch_size: 32
  optimizer: adam
  learning_rate: 0.001
  protocol_params: {}
```

- Protocol choices: `FedAvg`, `FedProx`, `FedOpt`, and `WebDISCO-style`.
- `local_steps` is the exact number of optimizer updates performed by a
  selected client per communication round.
- `batch_mode` is `mini_batch` or `full_batch`; full batch is an explicit
  sensitivity setting, not the default.
- Optimizers are `adam` and `sgd`.
- FedProx accepts `protocol_params: {mu: 0.01}`.
- FedOpt accepts parameters such as `optimizer`, `server_lr`, `beta1`,
  `beta2`, and `tau`.
- WebDISCO-style accepts `max_iter`, `tolerance`, `l2`, `ridge`, and
  `max_step_norm`; its adapter enforces CoxPH and full-client constraints.

Invalid model-protocol combinations fail before training through the protocol
adapter's compatibility checks.

## `references`

```yaml
references:
  centralized: true
  local_models: true
  aggregate_local: sample_weighted
  optimizer_steps: 50
```

`centralized` adds a Center reference trained on the full training split.
`local_models` adds every isolated client plus a sample-size-weighted local
summary. `optimizer_steps` makes the Center/Local training budget explicit;
omit it to use `global_rounds * local_steps`.

## `evaluation`

```yaml
evaluation:
  metrics: [c_index, ibs]
  save_round_metrics: true
  save_predictions: true
  time_grid_quantiles: [0.05, 0.95]
```

The supported metrics are time-dependent C-index and integrated Brier score
(IBS). Round metrics are needed for learning-trajectory plots. Saved survival
predictions are the federated model's held-out test predictions in long form.

## `visualization`

```yaml
visualization:
  enabled: true
  style: publication
  formats: [png, pdf]
```

Styles are `publication` and `presentation`; formats are `png`, `pdf`, and
`svg`.

Client-partition plots:

```yaml
client_partition:
  type: strip
  show_sample_size: true
  show_censoring_rate: true
```

Types are `strip`, `scatter`, `boxplot`, `violin`, and `censoring_bar`.

Round-metric plots:

```yaml
round_metrics:
  type: line
  aggregate_seeds: mean_ci95
  separate_panels: true
```

Seed aggregation can be `none`, `mean_sd`, or `mean_ci95`.

Final-metric plots:

```yaml
final_metrics:
  type: boxplot
  metrics: [c_index, ibs]
  methods: [Center, Federated, Weighted Local]
  show_raw_points: true
  show_mean: true
```

Types are `boxplot`, `violin`, `dotplot`, `bar`, and `table`. C-index and IBS
are placed in separate panels because their scales and desirable directions
differ. A black diamond marks the mean when `show_mean` is enabled.

Set `composite: {enabled: true, preset: overview}` for an optional overview
with the client distribution, configuration, round-wise C-index, and separate
C-index/IBS result panels. This layout is independent of any dataset or paper.

## `output`

```yaml
output:
  directory: results/cohort_comparison
  overwrite: false
```

The default is fail-safe: a non-empty output directory is rejected. Set
`overwrite: true` only when replacing a previous run is intentional.

The output includes `resolved_config.yaml`, `environment.json`,
`data_summary.json`, client partition tables, `raw_results.csv`,
`summary_metrics.csv`, optional round metrics and predictions,
`failures.csv`, `run_status.json`, `run_manifest.json`, and the requested
figures. A successful run has `complete: true`, no failed seeds, and no
non-finite metric rows.
