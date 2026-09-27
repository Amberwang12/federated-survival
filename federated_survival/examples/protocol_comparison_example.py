"""Small executable example for the four built-in protocol adapters."""

from federated_survival import DataGenerator, FederatedSurvival, SimulationConfig


frame = DataGenerator(
    SimulationConfig(n_samples=240, n_features=5, random_state=42)
).generate("weibull", c_mean=0.4)

experiments = [
    ("DeepSurv", "FedAvg", {}),
    ("DeepSurv", "FedProx", {"mu": 0.01}),
    ("DeepSurv", "FedOpt", {"optimizer": "adam", "server_lr": 0.01}),
    ("CoxPH", "WebDISCO-style", {"max_iter": 50, "tolerance": 1e-7}),
]

for model_name, protocol_name, protocol_params in experiments:
    estimator = FederatedSurvival(
        model=model_name,
        protocol=protocol_name,
        protocol_params=protocol_params,
        n_clients=5,
        global_rounds=3,
        local_steps=2,
        batch_size=32,
        random_state=42,
        weight_decay=0.0,
    ).fit(frame)
    summary = estimator.get_run_summary()
    print(
        model_name,
        protocol_name,
        summary["final_metrics"],
    )

