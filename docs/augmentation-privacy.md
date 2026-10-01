# Augmentation and experimental privacy calls

This page extends the [composable Python workflow](interactive-experiments.md).
Run these calls with the package's Python environment. The real datasets
referenced below are bundled with the package and load through
`fs.load_real_data`, so no repository-relative paths are needed.
With PyCharm, set `MPLBACKEND=Agg` in the Console or Run configuration **before
starting** it if the macOS interactive Matplotlib backend fails. Set
`show=False` to save figures without opening a window.

## Compare the two augmentation methods visually

`MVAEC` trains a generator at each client and adds its generated observations
to that client. `MVAES` pools generated observations and redistributes them;
therefore, it is **not** a no-sharing workflow. Both methods leave the held-out
test set unchanged. The returned dictionaries map each client ID to `(X, y)`
arrays containing the original observations first, followed by synthetic
observations. The internal MVAE trainer currently runs 500 epochs per client.

```python
import matplotlib.pyplot as plt
import federated_survival as fs

data = fs.load_real_data("gbsg")  # bundled GBSG table, ships with the package
split = fs.partition_data(
    data, n_clients=3, method="censoring-non-iid", test_size=0.2,
    seed=0, standardize=True, missing_values="error",
)

local = fs.augment_clients(
    split, method="MVAEC", k=0.3,
    sampling="sparse", sparse_gamma=0.1, seed=42,
)
pooled = fs.augment_clients(
    split, method="MVAES", k=0.3,
    sampling="sparse", sparse_gamma=0.1, seed=42,
)
print(fs.summarize_augmentation(split, local).to_string(index=False))
figure = fs.plot_augmentation_comparison(
    split, {"MVAEC": local, "MVAES": pooled},
    chart="strip",  # x: Client 1/2/3; y: time
    output_path="results/gbsg_augmentation_comparison.png",
    show=False,
)
plt.close(figure)
```

This saves one two-panel client-wise scatter plot. Each panel has Client 1–3
on the x-axis and time on the y-axis; colour/marker distinguish original
uncensored, original censored, and synthetic uncensored observations. For a
feature–time scatter instead, call the same function with
`chart="scatter", x="feature_2", y="time"` and a different output path.
`feature_2` is age in this GBSG file, standardized using training data.
Here `k=0.3`
requests `int(0.3 * number_of_uncensored_client_observations)` synthetic rows
per client, **not** 30% of all client observations. The returned `local` and
`pooled` arrays stay in memory; no CSV is written automatically.

With the default `sampling="sparse"`, the generator follows the original
FSA-MVAE targeting rule separately for every client:

1. retain uncensored observations only;
2. split their observed times at the median;
3. compare the time spans of the earlier and later halves;
4. treat the half with the wider span as the sparse region (a tie selects the
   later half);
5. perturb latent codes belonging to that half by Gaussian noise with standard
   deviation `sparse_gamma`, then decode exactly the number requested by `k`.

This is a **targeting strategy**, not a range constraint: a decoded synthetic
time can still fall outside the selected half. The current plot must therefore
be inspected to determine where generated samples actually landed. The
`summarize_augmentation` table reports `synthetic_in_target_fraction` for that
check. Set
`sampling="unconditional"` to sample latent vectors from the standard normal
prior instead. Increasing `sparse_gamma` explores farther away from the source
latent codes, but does not guarantee later times and can reduce realism.

The scatter plot is a **quality diagnostic**, not proof of realistic or valid
augmentation. In the current GBSG check, unconditional samples cluster much
earlier than the original observations, while sparse-targeted samples mostly
enter the selected later half. That improvement does not by itself establish
that the joint feature/time distribution is realistic. Inspect survival-time
and feature distributions, impossible categorical values, and downstream
sensitivity before interpreting augmented experiments.

To train a single federated model with internal augmentation instead of just
inspecting the generated arrays:

```python
fit = fs.FederatedSurvival(
    model="DeepSurv", protocol="FedAvg", n_clients=3,
    global_rounds=2, local_steps=1, random_state=0, k=0.3,
    augmentation_sampling="sparse", augmentation_sparse_gamma=0.1,
).fit(split, augmentation="MVAEC")  # or "MVAES"
print(fit.get_run_summary())
```

`fit(..., augmentation=...)` generates a new augmented dataset internally; it
does not reuse the arrays produced by the plotting calls. The paired
`compare_methods` API deliberately runs without internal augmentation or
privacy perturbation and should not be described as producing augmented
Center/Federated/Local comparisons.

## Tuned augmentation configurations

Whether augmentation helps is decided by the MVAE generator settings, not only
by `k` and `sparse_gamma`. The default `latent_num=10` is larger than the
feature count of the bundled examples and produces noisy synthetic rows;
shrinking the latent space is the single most impactful change. The
configurations below were validated with five partition seeds (0–4) on the
same dataset, with DeepSurv/FedAvg and identical `global_rounds × local_steps`
budgets for the plain and the augmented runs.

| Setting | GBSG censoring-non-IID, 3 clients | GBSG Dirichlet (alpha=0.5), 5 clients | Weibull simulation, time-non-IID, 3 clients |
| --- | --- | --- | --- |
| `global_rounds` / `local_steps` | 30 / 1 | 30 / 5 | 30 / 1 |
| `learning_rate` | 0.003 | 0.001 | 0.003 |
| `augmentation_sampling` | sparse | sparse | unconditional |
| `k` | 0.2 | 0.5 | 0.3 |
| `latent_num` | 5 | 10 | 5 |
| `beta` | 1 | 5 | 5 |
| Plain federated C-index (mean ± SD) | 0.609 ± 0.030 | 0.628 ± 0.030 | 0.580 ± 0.028 |
| Augmented federated C-index (mean ± SD) | 0.637 ± 0.014 | 0.645 ± 0.025 | 0.600 ± 0.024 |
| Centralized (Center) C-index | 0.634 | 0.645 | 0.581 |
| Weighted Local C-index | 0.591 | 0.591 | 0.541 |

Across these five-seed runs the augmented federated model beats the plain
federated model in 4/5, 5/5, and 4/5 seeds respectively, matches or exceeds
the centralized model on average, and both federated views clearly beat
per-client local training. Example call for the GBSG censoring-non-IID
setting:

```python
fit = fs.FederatedSurvival(
    model="DeepSurv", protocol="FedAvg", n_clients=3,
    global_rounds=30, local_steps=1, batch_size=32, learning_rate=0.003,
    random_state=42, k=0.2, latent_num=5, beta=1,
    augmentation_sampling="sparse", augmentation_sparse_gamma=0.1,
).fit(split, augmentation="MVAEC")
print(fit.get_run_summary()["final_metrics"])
```

Tuning guidance from the same sweep: reduce `latent_num` towards the feature
count, raise `beta` when synthetic event times look unrealistic, prefer
`unconditional` sampling when clients cover narrow time ranges
(time-non-IID), and keep `sparse_gamma` small (0.1 and 0.3 performed nearly
identically). These are empirical results for the bundled datasets, not
universal recommendations; re-validate on your own data before claiming an
augmentation benefit.

## Configure experimental update perturbation

The high-level estimator passes `FSAConfig` privacy fields as keyword
arguments. For a Gaussian-noise *demonstration* with FedProx:

```python
fit = fs.FederatedSurvival(
    model="DeepSurv", protocol="FedProx",
    protocol_params={"mu": 0.01},
    n_clients=3, global_rounds=2, local_steps=1, random_state=0,
    use_differential_privacy=True,
    dp_mechanism="gaussian",
    dp_epsilon=1.0,
    dp_delta=1e-5,
    dp_sensitivity=1.0,
    dp_noise_multiplier=0.05,
    dp_clip_norm=1.0,
).fit(split)
print(fit.get_run_summary()["final_metrics"])
print(fit.get_privacy_info())
```

This example reports test/train C-index and IBS and the **configured**
perturbation settings. `get_privacy_info()` deliberately says
`formal_accounting_available: False`; it does not report an achieved privacy
budget. The numeric values above are smoke-test settings, not privacy or
utility recommendations.

| Field | Current role |
| --- | --- |
| `use_differential_privacy` | switches on experimental perturbation of each complete client model update |
| `dp_mechanism="gaussian"` | noise with scale driven by `dp_noise_multiplier` and sensitivity |
| `dp_noise_multiplier` | Gaussian noise multiplier; increasing it increases noise |
| `dp_clip_norm` | clips a complete client update, not each person's gradient |
| `dp_sensitivity` | scale input for the perturbation mechanism |
| `dp_epsilon`, `dp_delta` | legacy/configuration fields; **not** an achieved multi-round guarantee for Gaussian training |
| `dp_mechanism="laplace"` | alternative update perturbation; `dp_epsilon` influences its noise scale, but no end-to-end accountant exists |

The utility's exponential mechanism selects discrete candidates. It cannot
be used as elementwise noise for continuous model updates in this training
path. The current package does not provide per-example gradient clipping,
privacy amplification, multi-round privacy accounting, or secure aggregation.
Do not claim record-level `(epsilon, delta)`-DP or use these settings alone as
protection for sensitive clinical data. See [Privacy scope](privacy.md).

Neither the paired `compare_methods(...)` call nor the schema-v1 YAML
experiment interface currently exposes augmentation or privacy settings as
part of its audited Center/Federated/Local workflow. Use the high-level
estimator above for exploratory runs and keep these results separate from
paired baseline figures.
