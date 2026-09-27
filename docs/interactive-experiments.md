# Composable experiments in Python

Use the function API when you want to inspect intermediate data or combine
specific models, protocols, and plots in a PyCharm Python Console or notebook.
YAML remains available for recorded, multi-seed runs with manifests.

## One model and protocol

```python
import federated_survival as fs

data = fs.simulate_data(
    mechanism="weibull", n_samples=300, n_features=5, seed=42
)
split = fs.partition_data(
    data, n_clients=3, method="iid", test_size=0.2, seed=42
)
fs.plot_partition(
    split, chart="strip", output_path="results/partition.png", show=False
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
print(result.round_metrics[["round", "test_c_index", "test_ibs"]])
result.plot(kind="round_metrics", output_path="results/round_metrics.png")
result.plot(
    kind="final_metrics", chart="dotplot", output_path="results/final_metrics.png"
)
```

`result.metrics` contains Center, Federated, every Local-client, and their
sample-weighted Local average. `result.round_metrics` contains federated
round-wise diagnostics. All methods use the same held-out test set and input
partition. `reference_steps` is the optimizer-update budget for Center and
Local; when omitted it defaults to `global_rounds * local_steps`.

For real data, replace the first call with:

```python
data = fs.load_data(
    "data.csv", duration_column="time", event_column="status"
)
split = fs.partition_data(
    data, n_clients=3, method="iid", test_size=0.2,
    seed=42, standardize=True, missing_values="error",
)
```

For an existing GBSG dataset, the same functions work without a dedicated
GBSG script or configuration:

```python
data = fs.load_data(
    "data/real/gbsg.csv", duration_column="time", event_column="status"
)
splits = fs.partition_data_many(
    data, seeds=[0, 1, 2, 3, 4], n_clients=3,
    method="iid", test_size=0.2, standardize=True,
)
result = fs.compare_methods(
    splits, model="DeepSurv", protocol="FedAvg",
    global_rounds=10, local_steps=5, reference_steps=50,
)
print(result.metrics[["seed", "method", "c_index", "ibs"]])
figures = result.save_figures(
    "results/gbsg_five_seeds",
    final_chart="boxplot",
    combine=True,
)
print(figures)
```

For a separate mean/SD table from the same run, call
`result.plot(kind="final_metrics", chart="table", show=False,
output_path="results/gbsg_five_seeds/final_metrics_table.png")` and close the
returned Matplotlib figure. The table's SD is the sample standard deviation
across seeds; a one-seed table shows `--` because there is no replicate SD.
The per-seed numerical results are available as `result.metrics` and can be
saved with `result.metrics.to_csv("results/gbsg_five_seeds/raw_metrics.csv",
index=False)`.

`standardize=True` uses training-set statistics independently within each
seed; `missing_values` can be `error`, `drop`, or `median` (training-set
medians). Input paths in the Python API are relative to the current working
directory. These example parameters demonstrate the workflow; exact published
numbers depend on matching the original settings and numerical environment.

## Multiple models and protocols

```python
result = fs.compare_experiments(
    split,
    models=["CoxPH", "DeepSurv"],
    protocols=["FedAvg", "FedProx"],
    protocol_params={"FedProx": {"mu": 0.01}},
    global_rounds=10,
    local_steps=5,
    seed=42,
)

print(result.metrics[["model", "protocol", "method", "c_index", "ibs"]])
result.plot(
    kind="final_metrics",
    chart="bar",
    model="DeepSurv",
    protocol="FedProx",
    output_path="results/deepsurv_fedprox.png",
)
```

The function runs the cross-product of the selected models and protocols.
Compatibility is checked before training begins; for example, WebDISCO-style
supports linear CoxPH only. Supported models and protocols can be inspected
with `fs.available_model_adapters()` and `fs.available_federated_protocols()`.

## Multiple seeds and boxplots

For repeated experiments, first create one partition per seed. Each model and
method within a seed shares that seed's held-out test set:

```python
splits = fs.partition_data_many(
    data,
    seeds=[0, 1, 2, 3, 4],
    n_clients=3,
    method="iid",
)
result = fs.compare_methods(
    splits,
    model="DeepSurv",
    protocol="FedAvg",
    global_rounds=10,
    local_steps=5,
)
result.plot(
    kind="final_metrics",
    chart="boxplot",
    output_path="results/c_index_ibs_boxplots.png",
)
```

The returned table has a `seed` column. At least two seeds are required for
`boxplot` or `violin`; a single observation is not drawn as a box. The original
simulated dataset remains fixed in this example while train/test partition,
client allocation, and model initialization vary by seed.

## Plot and result choices

- `result.plot(kind="partition")`: client-wise observed-time strip plot with
  uncensored and censored observations distinguished;
- `result.plot(kind="partition", chart="scatter", x="feature_1", y="time")`:
  client-coloured feature scatter plot;
- `result.plot(kind="round_metrics")`: federated C-index and IBS over rounds;
- `result.plot(kind="final_metrics", chart="dotplot")`: final metrics for the
  three aggregate methods;
- final-metric `chart` can be `dotplot`, `bar`, or `table` for one seed;
  `boxplot` and `violin` are also available when there are at least two seeds.

`result.plot(...)` returns a Matplotlib Figure. Use `show=False` in a headless
script and close the returned figure when finished. Set `output_path` to save
PNG, PDF, or SVG according to its extension. To save tabular output, call
`result.metrics.to_csv(...)` or `result.round_metrics.to_csv(...)`.

`result.save_figures(output_dir, final_chart="boxplot", combine=False)` saves
three separate figures: client partition, two-panel round metrics, and
two-panel final metrics. Set `combine=True` to additionally save a generic
five-panel overview containing the same plots. The method returns the output
paths. It rejects existing target files by default; use `overwrite=True` only
when replacing them intentionally. The partition figure shows the first seed
unless `seed=` selects another.

The interactive API does not write a resolved configuration, environment
record, or run manifest. For a publication run needing those audit artifacts,
use the YAML workflow in [Configuration-driven experiments](configuration.md).
For MVAEC/MVAES scatter comparisons or the experimental client-update
perturbation settings, see [Augmentation and privacy calls](augmentation-privacy.md).
