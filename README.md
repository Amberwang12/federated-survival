# federated-survival

`federated-survival` is an open-source Python package for simulated horizontal
federated survival analysis. It provides composable functions for data input,
client partitioning, survival-model selection, federated-protocol selection,
Center/Federated/Local comparison, optional data augmentation, evaluation, and
publication-style visualization.

Model behavior and federated execution are separated through adapter
registries. A user can therefore combine supported data, model, protocol, and
plotting choices without writing a dataset-specific training script.

Raw records remain local during the simulated federated workflow. This alone
is not a formal differential-privacy, secure-aggregation, transport-security,
or production-deployment guarantee. See [Privacy scope](docs/privacy.md).

**Documentation:** <https://amberwang12.github.io/federated-survival/>

## What the package can do

| Capability | Public call | Main choices | Returned result |
| --- | --- | --- | --- |
| Simulate a survival table | `fs.simulate_data(...)` | Weibull, log-normal, SDGM1–4 | pandas survival table |
| Load and normalize a survival table | `fs.load_data(...)` | CSV or Excel | canonical feature/time/status table |
| Partition into train/test and clients | `fs.partition_data(...)` | IID, random, censoring-non-IID, time-non-IID, Dirichlet | train/test data and client arrays |
| Build paired partitions per seed | `fs.partition_data_many(...)` | one partition per seed | seed-to-partition mapping |
| Compare Center/Federated/Local for one model–protocol pair | `fs.compare_methods(...)` | one model and one protocol | Center, Federated, Local-client, and Weighted Local results |
| Compare a model × protocol grid | `fs.compare_experiments(...)` | multiple models and protocols | compatible model-protocol cells |
| Generate synthetic client rows | `fs.augment_clients(...)` | MVAEC or MVAES; sparse-targeted or unconditional sampling | original-plus-synthetic client arrays |
| Audit synthetic-target hit rate | `fs.summarize_augmentation(...)` | per-client sparse target and landing fraction | pandas summary table |
| Visualize the client partition | `fs.plot_partition(...)` | strip, scatter; separate or combined | Matplotlib figures and image files |
| Visualize final metrics and round history | `result.plot(...)`, `result.save_figures(...)` | line, dot, bar, table, box, violin; separate or combined | Matplotlib figures and image files |
| Run an audited end-to-end experiment from a config | `federated-survival run --config ...` | JSON/YAML schema-v1 configuration | metrics, predictions, figures, environment record, manifest |

The public registries can be inspected at runtime:

```python
import federated_survival as fs

print(fs.available_model_adapters())
print(fs.available_federated_protocols())
```

## Choice reference

### Data input

Real-data input accepts CSV and Excel files. Feature columns may have any
names; the duration and event columns are specified explicitly. Event values
must use `1` for an observed event and `0` for censoring.

```python
data = fs.load_data(
    "my_survival_data.csv",
    duration_column="time",
    event_column="status",
)
```

Simulated-data mechanisms are:

| `mechanism` | Meaning | Censoring control |
| --- | --- | --- |
| `weibull` | Weibull accelerated failure-time simulation | `censoring` |
| `lognormal` | log-normal accelerated failure-time simulation | `censoring` |
| `SDGM1` | proportional-hazards simulation | `censoring` |
| `SDGM2` | mild proportional-hazards violation | `u_max` |
| `SDGM3` | stronger proportional-hazards violation | `u_max` |
| `SDGM4` | log-normal-error simulation with covariate-dependent censoring | `c_step` |

For Weibull, log-normal, and SDGM1, `censoring=0.4` controls the censoring-time
distribution and normally produces a censoring fraction near 0.4 under the
default design. It does **not** force exactly 40% of rows to be censored. Always
report the realized value:

```python
print((data["status"] == 0).mean())
```

### Client partitioning

| `method` | Meaning | Typical use |
| --- | --- | --- |
| `iid` | status-stratified, approximately balanced client split | standard baseline |
| `random` | unstratified random split; IID only in expectation | simple random-site simulation |
| `non-iid` or `censoring-non-iid` | clients receive deliberately different censoring/event proportions | censoring heterogeneity |
| `time-non-iid` | clients receive different observed-time ranges | temporal heterogeneity |
| `dirichlet` | time-bin/status pseudo-classes are allocated with a Dirichlet distribution | combined heterogeneity |

`alpha` applies to `dirichlet`: smaller values create stronger heterogeneity.
The train/test split is status-stratified before the training rows are
allocated to clients. Every generated client partition is checked to contain
at least one observed event.

### Survival models

| `model` | Meaning | Important setting |
| --- | --- | --- |
| `CoxPH` | linear proportional-hazards Cox model | use `num_nodes=()` for the linear form |
| `DeepSurv` | neural proportional-hazards Cox model | `hidden_nodes`, `dropout` |
| `CoxCC` | case-control approximation to the Cox objective | neural-network settings |
| `CoxTime` | neural Cox model with time-dependent effects | neural-network settings |
| `LogisticHazard` | discrete-time logistic hazard model | `num_durations` |
| `PC-Hazard` | piecewise-constant hazard model | `num_durations` |
| `DeepHit` | single-event discrete-time DeepHit model | `num_durations` |

The current package implements the single-event survival setting. It does not
expose a competing-risks DeepHit workflow.

### Federated protocols

| `protocol` | Meaning | Compatibility or key parameter |
| --- | --- | --- |
| `FedAvg` | sample-weighted parameter averaging | all seven model adapters |
| `FedProx` | FedAvg plus a proximal local objective | all seven models; `protocol_params={"mu": ...}` |
| `FedOpt` | adaptive server-side optimization | all seven models; Adam, Yogi, or Adagrad |
| `WebDISCO-style` | aggregated Cox risk-set/event statistics | linear `CoxPH` only, full participation |

The WebDISCO-style adapter is an independent distributed Breslow Cox
implementation, not the original WebDISCO web service. Compatibility checks
reject unsupported combinations before training starts.

### Comparisons and evaluation

`fs.compare_methods(...)` evaluates four views on the same held-out test set:

- `Center`: one model trained on the pooled training rows;
- `Federated`: the selected federated protocol;
- `Local-client`: one independent model for each client;
- `Weighted Local`: client-local metrics combined by client sample size.

The common metrics are:

| Metric | Interpretation |
| --- | --- |
| C-index | discrimination/ranking; higher is better |
| IBS | integrated prediction error; lower is better |

### Data augmentation

| Choice | Meaning |
| --- | --- |
| `method="MVAEC"` | each client trains and uses its own MVAE generator |
| `method="MVAES"` | generated rows are pooled and redistributed; this is not a no-sharing workflow |
| `sampling="sparse"` | condition on latent codes from the wider median half of uncensored times |
| `sampling="unconditional"` | sample latent vectors from the standard normal prior |

`k=0.3` generates `int(0.3 * uncensored_client_rows)` synthetic rows per
client; it is not 30% of all rows. Sparse sampling targets a region but does
not mathematically force every decoded time to remain there. Use
`fs.summarize_augmentation(...)` to measure the actual fraction.

### Visualization

- `fs.plot_partition(..., chart="strip")`: clients on the x-axis and observed
  time on the y-axis, with censored/uncensored markers;
- `fs.plot_partition(..., chart="scatter")`: feature-versus-time plot;
- `result.plot(kind="round_metrics")`: C-index and IBS by communication round;
- `result.plot(kind="final_metrics", chart="dotplot" | "bar" | "table")`:
  one-seed final results;
- `result.plot(kind="final_metrics", chart="boxplot" | "violin")`:
  repeated-seed results;
- `result.save_figures(..., combine=True)`: save separate plots and an
  additional combined overview.

Box and violin plots require at least two seeds. Five seeds are useful for an
initial check; publication analyses commonly use more repetitions.

## Installation

Install the published package:

```bash
python -m pip install federated-survival
```

Install a cloned source tree for development:

```bash
python -m pip install -e ".[dev,docs]"
```

Verify the selected environment and package registries:

```bash
federated-survival doctor
```

## Example 1: simulated data

This complete function-based example generates data, creates three
time-non-IID clients, runs DeepSurv with FedAvg, prints all comparison metrics,
and saves the standard plots.

```python
from pathlib import Path
import federated_survival as fs

output = Path("results/simulation_example")
data = fs.simulate_data(
    mechanism="weibull",
    n_samples=600,
    n_features=5,
    censoring=0.4,
    seed=42,
)
print("Realized censoring fraction:", (data["status"] == 0).mean())

split = fs.partition_data(
    data,
    n_clients=3,
    method="time-non-iid",
    test_size=0.2,
    seed=42,
    standardize=True,
)
result = fs.compare_methods(
    split,
    model="DeepSurv",
    protocol="FedAvg",
    global_rounds=10,
    local_steps=5,
    reference_steps=50,
    batch_size=32,
    seed=42,
)
print(result.metrics[["method", "client", "c_index", "ibs"]])
paths = result.save_figures(
    output,
    final_chart="dotplot",
    combine=True,
    overwrite=True,
)
print(paths)
```

To inspect sparse-targeted augmentation before training:

```python
augmented = fs.augment_clients(
    split, method="MVAEC", k=0.3,
    sampling="sparse", sparse_gamma=0.1, seed=42,
)
print(fs.summarize_augmentation(split, augmented))
fs.plot_augmentation_comparison(
    split,
    {"MVAEC sparse": augmented},
    chart="strip",
    output_path=output / "augmentation.png",
    show=False,
)
```

## Example 2: real CSV data

Prepare a CSV with feature columns plus duration and event columns, for
example:

```text
age,tumor_size,marker,time,status
58,2.1,0.72,814,1
67,3.0,0.41,391,0
```

The following example performs five paired repetitions, creates
censoring-non-IID clients, runs DeepSurv with FedProx, and produces C-index and
IBS boxplots. Replace the path and column names with your own data definition.

```python
from pathlib import Path
import federated_survival as fs

output = Path("results/real_data_example")
data = fs.load_data(
    "my_survival_data.csv",
    duration_column="time",
    event_column="status",
)
splits = fs.partition_data_many(
    data,
    seeds=[0, 1, 2, 3, 4],
    n_clients=3,
    method="censoring-non-iid",
    test_size=0.2,
    standardize=True,
    missing_values="error",
)
result = fs.compare_methods(
    splits,
    model="DeepSurv",
    protocol="FedProx",
    protocol_params={"mu": 0.01},
    global_rounds=10,
    local_steps=5,
    reference_steps=50,
    batch_size=32,
    learning_rate=0.001,
)
print(result.metrics[["seed", "method", "client", "c_index", "ibs"]])
paths = result.save_figures(
    output,
    final_chart="boxplot",
    combine=True,
    overwrite=True,
)
result.metrics.to_csv(output / "raw_metrics.csv", index=False)
print(paths)
```

Input files remain user-supplied; dataset-specific result files are not
included in the installed package.

## Three execution interfaces

1. **Composable Python functions** are intended for the PyCharm Python
   Console, notebooks, and scripts where intermediate objects should remain
   inspectable.
2. **`FederatedSurvival` estimator** provides one high-level fit/predict object
   and is the entry point for internal augmentation or experimental update
   perturbation.
3. **JSON/YAML workflow** is intended for recorded multi-seed experiments and
   writes resolved configuration, environment metadata, failures, status, and
   artifact hashes.

Example CLI calls:

```bash
federated-survival doctor --output doctor.json
federated-survival config-options
federated-survival quickstart --model DeepSurv --protocol FedAvg --output results/quickstart
federated-survival run --config federated_survival/examples/seven_model_smoke.json
```

See [Configuration-driven experiments](docs/configuration.md) for every
schema-v1 field. Paths inside YAML are resolved relative to the YAML file.

## Documentation

The documentation is published at
<https://amberwang12.github.io/federated-survival/>. The same pages ship with
the source tree, where they can be read without a network connection:

- [Composable Python experiments](docs/interactive-experiments.md)
- [Augmentation and privacy calls](docs/augmentation-privacy.md)
- [Configuration reference](docs/configuration.md)
- [Model adapters](docs/model-adapters.md)
- [Federated protocols](docs/federated-protocols.md)
- [API reference](docs/api.md)
- [Reproducibility](docs/reproducibility.md)
- [Release checklist](docs/releasing.md)
- [Package structure](PACKAGE_STRUCTURE.md)

## Testing

```bash
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python -m pytest -q
```

Tests are part of the source repository and source distribution because they
document and verify expected behavior. Generated `results/`, virtual
environments, IDE files, caches, and validation XML reports are development
artifacts and are not package runtime data.

## Extending the package

New survival learners implement `ModelAdapter` and register with
`register_model_adapter`. New federated strategies implement
`FederatedProtocol`, declare their capabilities, and register with
`register_federated_protocol`. The data, evaluation, plotting, and experiment
recording workflow does not need to be rewritten.

## Software scope

This is a research and evaluation environment for simulated horizontal
federated execution. It does not provide a cross-institutional transport
layer, authentication, secure aggregation, or a formal record-level
multi-round differential-privacy guarantee. Production use with sensitive
data requires a separately validated deployment architecture and threat model.

## License

MIT License. See [LICENSE](LICENSE).
