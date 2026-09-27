# Quickstart

The high-level API infers the feature dimension and owns data splitting,
training, prediction, validation, and evaluation.

```python
from federated_survival import DataGenerator, FederatedSurvival, SimulationConfig

data = DataGenerator(
    SimulationConfig(n_samples=300, n_features=10, random_state=42)
).generate("weibull", c_mean=0.4)

model = FederatedSurvival(
    model="DeepSurv",
    n_clients=5,
    global_rounds=10,
    local_steps=2,
    batch_size=32,
).fit(data)

survival = model.predict_survival(model.dataset_.test_data)
summary = model.get_run_summary()
```

For data that already contain an institution identifier:

```python
model.fit(
    dataframe,
    duration_col="time",
    event_col="status",
    client_col="hospital",
)
```

The command-line equivalent is:

```bash
federated-survival quickstart --model DeepSurv --output results/quickstart
```

The output directory contains round metrics, survival predictions, a summary,
and `run_manifest.json` with package versions and SHA-256 artifact hashes.
