from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import federated_survival as fs


def _dataset():
    frame = fs.simulate_data(n_samples=120, n_features=3, seed=17)
    return fs.partition_data(frame, n_clients=3, method="iid", seed=17)


def test_compare_methods_runs_paired_references_and_selectable_plots(tmp_path):
    dataset = _dataset()
    result = fs.compare_methods(
        dataset,
        model="DeepSurv",
        protocol="FedAvg",
        global_rounds=2,
        local_steps=1,
        reference_steps=2,
        batch_size=16,
        hidden_nodes=(8,),
        seed=17,
    )

    assert result.dataset is dataset
    assert set(result.metrics["method"]) == {
        "Center",
        "Federated",
        "Local-client",
        "Weighted Local",
    }
    assert len(result.metrics[result.metrics["method"].eq("Local-client")]) == 3
    assert result.metrics[["c_index", "ibs"]].notna().all().all()
    assert len(result.round_metrics) == 2
    assert result.predictions.empty

    for kind in ("partition", "round_metrics", "final_metrics"):
        output = tmp_path / (kind + ".png")
        figure = result.plot(kind=kind, output_path=output, show=False)
        assert output.exists()
        assert len(figure.axes) == (1 if kind == "partition" else 2)
        if kind == "partition":
            assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
                "Client 1",
                "Client 2",
                "Client 3",
            ]
        plt.close(figure)
    scatter = result.plot(kind="partition", chart="scatter", show=False)
    assert scatter.axes[0].get_xlabel() == "feature 1"
    plt.close(scatter)
    with pytest.raises(ValueError, match="at least two paired seeds"):
        result.plot(kind="final_metrics", chart="boxplot", show=False)
    with pytest.raises(ValueError, match="at least two paired seeds"):
        result.save_figures(tmp_path / "single_boxplot", final_chart="boxplot")
    assert not (tmp_path / "single_boxplot").exists()


def test_compare_experiments_reuses_one_partition_and_labels_cells(monkeypatch):
    dataset = _dataset()
    calls = []

    def fake_baselines(config, received, seed, **kwargs):
        calls.append((config.model_type, config.federated_protocol, received, seed))
        kwargs["telemetry_rows"].append(
            {
                "seed": seed,
                "model": config.model_type,
                "round": 1,
                "test_c_index": 0.6,
                "test_ibs": 0.2,
            }
        )
        return pd.DataFrame(
            [
                {
                    "seed": seed,
                    "model": config.model_type,
                    "method": method,
                    "client": "all",
                    "n_train": len(received.train_data),
                    "c_index": 0.6,
                    "ibs": 0.2,
                }
                for method in ("Center", "FSA", "Local")
            ]
        )

    monkeypatch.setattr("federated_survival.experiment_api.run_paired_baselines", fake_baselines)
    result = fs.compare_experiments(
        dataset,
        models=["DeepSurv", "CoxPH"],
        protocols=["FedAvg", "FedProx"],
        global_rounds=1,
        local_steps=1,
        seed=17,
        protocol_params={"FedProx": {"mu": 0.02}},
    )

    assert len(calls) == 4
    assert all(received is dataset and seed == 17 for _, _, received, seed in calls)
    assert len(result.metrics) == 12
    assert len(result.round_metrics) == 4
    assert result.configurations[("DeepSurv", "FedProx")].proximal_mu == 0.02
    with pytest.raises(ValueError, match="select both model and protocol"):
        result.plot(show=False)
    figure = result.plot(model="CoxPH", protocol="FedAvg", kind="final_metrics", show=False)
    plt.close(figure)


def test_incompatible_cells_fail_before_any_training(monkeypatch):
    dataset = _dataset()

    def must_not_train(*args, **kwargs):
        pytest.fail("compatibility validation must precede training")

    monkeypatch.setattr("federated_survival.experiment_api.run_paired_baselines", must_not_train)
    with pytest.raises(ValueError, match="supports only"):
        fs.compare_experiments(
            dataset,
            models=["CoxPH", "DeepSurv"],
            protocols=["WebDISCO-style"],
            global_rounds=1,
            local_steps=1,
        )


def test_protocol_parameters_are_checked_before_training():
    with pytest.raises(ValueError, match="unknown protocol parameter"):
        fs.compare_methods(
            _dataset(),
            protocol="FedProx",
            protocol_params={"unrecognized": 1},
        )


def test_multiple_seeds_enable_boxplots_without_yaml(monkeypatch, tmp_path):
    data = fs.simulate_data(n_samples=120, n_features=3, seed=21)
    datasets = fs.partition_data_many(data, seeds=[0, 1, 2], n_clients=3)
    seen = []

    def fake_baselines(config, received, seed, **kwargs):
        seen.append((seed, received))
        kwargs["telemetry_rows"].append(
            {
                "seed": seed,
                "model": config.model_type,
                "round": 1,
                "test_c_index": 0.6 + seed * 0.01,
                "test_ibs": 0.2 - seed * 0.01,
            }
        )
        return pd.DataFrame(
            [
                {
                    "seed": seed,
                    "model": config.model_type,
                    "method": method,
                    "client": "all",
                    "n_train": len(received.train_data),
                    "c_index": 0.6 + seed * 0.01,
                    "ibs": 0.2 - seed * 0.01,
                }
                for method in ("Center", "FSA", "Local")
            ]
        )

    monkeypatch.setattr("federated_survival.experiment_api.run_paired_baselines", fake_baselines)
    result = fs.compare_methods(datasets, global_rounds=1, local_steps=1)

    assert len(seen) == 3
    assert all(received is datasets[seed] for seed, received in seen)
    assert set(result.metrics["seed"]) == {0, 1, 2}
    output = tmp_path / "c_index_ibs_boxplots.png"
    figure = result.plot(chart="boxplot", output_path=output, show=False)
    assert output.exists()
    assert len(figure.axes) == 2
    plt.close(figure)

    separate = result.save_figures(tmp_path / "separate", final_chart="boxplot")
    assert set(separate) == {"partition", "round_metrics", "final_metrics"}
    assert all(path.exists() for path in separate.values())
    with pytest.raises(FileExistsError, match="figure files already exist"):
        result.save_figures(tmp_path / "separate", final_chart="boxplot")
    combined = result.save_figures(tmp_path / "combined", combine=True, final_chart="boxplot")
    assert set(combined) == {"partition", "round_metrics", "final_metrics", "combined"}
    assert all(path.exists() for path in combined.values())
    assert plt.imread(combined["combined"]).shape[0] > plt.imread(combined["combined"]).shape[1]
