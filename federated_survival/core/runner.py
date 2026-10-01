"""
Federated survival analysis runner.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from copy import deepcopy
from pycox.evaluation import EvalSurv

from .config import FSAConfig
from .server import Server
from .client import Client
from .augmenter import DataAugmenter
from ..utils.metrics import evaluation_time_grid
from ..models import get_model_adapter
from ..protocols import get_federated_protocol
from tqdm import tqdm


class FSARunner:
    """Runner for federated survival analysis."""

    def __init__(self, config: FSAConfig):
        """
        Initialize the federated survival analysis runner.

        Args:
            config: Federated learning configuration.
        """
        self.config = config
        self.adapter = get_model_adapter(config.model_type)
        self.protocol = get_federated_protocol(config.federated_protocol)
        self.protocol.validate_config(config)
        self.set_random_seed()

    def set_random_seed(self):
        """Set the random seed."""
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_seed)

    def _get_label_transform(self):
        """Get the label transformer."""
        self.adapter = get_model_adapter(self.config.model_type)
        return self.adapter.create_label_transform(self.config)

    def _get_model(self, net: nn.Module, labtrans=None):
        """Get the survival analysis model."""
        self.adapter = get_model_adapter(self.config.model_type)
        return self.adapter.build_model(
            net,
            self.config,
            label_transform=labtrans,
            optimizer=torch.optim.Adam,
        )

    def run(
        self, data: Any, type: str = "raw", aug_method: str = "MVAEC"
    ) -> Dict[str, List[float]]:
        """
        Run federated survival analysis.

        Args:
            data: Survival data with the following attributes:
                - clients_set: Client datasets in the form {client_id: (train_X, train_y)}
                - raw_aug_clients_set: Client datasets augmented from raw data,
                  in the form {client_id: (train_X, train_y)}
                - test_data: Test data
                - test_label: Test labels
            type: Data type, 'raw' or 'raw_aug'
            aug_method: Data augmentation method, 'MVAEC' or 'MVAES'

        Returns:
            dict: {'train_Cindex': train_Cindex, 'train_IBS': train_IBS, 'test_Cindex': test_Cindex, 'test_IBS': test_IBS}
        """
        self.adapter = get_model_adapter(self.config.model_type)

        # Get training data
        if type == "raw":
            clients_set = data.clients_set
        elif type == "raw_aug":
            augmenter = DataAugmenter(
                latent_num=self.config.latent_num,
                hidden_num=self.config.hidden_num,
                alpha=self.config.alpha,
                beta=self.config.beta,
                sampling=self.config.augmentation_sampling,
                sparse_gamma=self.config.augmentation_sparse_gamma,
            )
            if aug_method == "MVAEC":
                clients_set = augmenter.mvaec(data.clients_set, k=self.config.k)
            elif aug_method == "MVAES":
                clients_set = augmenter.mvaes(data.clients_set, k=self.config.k)
            else:
                raise ValueError(f"Unsupported augmentation method: {aug_method}")
        else:
            raise ValueError(f"Unsupported data type: {type}")

        if self.protocol.owns_training_loop:
            if type != "raw":
                raise ValueError(
                    f"protocol {self.protocol.name!r} does not support data augmentation"
                )
            return self._run_webdisco_style(data, clients_set)

        _y_train = np.empty(shape=(0, 2), dtype=np.float32)
        for i in clients_set.keys():
            _y_train = np.vstack((_y_train, clients_set[i][1]))
        y_train = self._get_target(_y_train)

        # Let the registered model adapter own label fitting and output shape.
        labtrans = self.adapter.configure_targets(self.config, *y_train)

        # Get test data
        durations_test, events_test = self._get_target(data.test_label)

        # Initialize server and clients
        server = Server(self.config)
        self.protocol.begin_run(self.config, server.global_model.state_dict())
        clients = []
        for i in clients_set.keys():
            clients.append(
                Client(
                    self.config,
                    server.global_model,
                    clients_set[i],
                    i,
                    protocol=self.protocol,
                )
            )

        # If differential privacy is enabled, print privacy protection info
        if self.config.use_differential_privacy:
            if self.config.verbose:
                mechanism = (
                    self.config.dp_mechanism if hasattr(self.config, "dp_mechanism") else "gaussian"
                )
                print("Differential privacy enabled:")
                print(f"  - mechanism: {mechanism.upper()}")
                print(f"  - privacy budget (epsilon): {self.config.dp_epsilon}")
                if mechanism == "gaussian":
                    print(f"  - failure probability (delta): {self.config.dp_delta}")
                    print(f"  - noise multiplier: {self.config.dp_noise_multiplier}")
                print(f"  - sensitivity: {self.config.dp_sensitivity}")
                if mechanism in ["gaussian", "laplace"]:
                    print(f"  - gradient clipping norm: {self.config.dp_clip_norm}")

        # Record metrics
        train_Cindex = []
        train_IBS = []
        test_Cindex = []
        test_IBS = []

        # Federated learning training
        train_loss = []
        update_direction_norm = []
        client_drift = []
        update_direction_dispersion = []
        selected_clients = []
        communication_bytes = []
        model_bytes = sum(
            value.numel() * value.element_size()
            for value in server.global_model.state_dict().values()
        )

        for e in tqdm(
            range(self.config.global_epochs),
            disable=not self.config.show_progress,
        ):
            # Select clients
            candidates = random.sample(
                clients, max(round(self.config.client_sample_ratio * len(clients)), 1)
            )

            # Compute the total number of samples
            N = sum(c.N for c in candidates)
            round_directions = []
            round_weights = []
            round_losses = []
            round_drifts = []
            round_states = []
            global_state = {
                name: value.detach().clone()
                for name, value in server.global_model.state_dict().items()
            }

            # Client training
            for i, c in enumerate(candidates, 1):
                if self.config.verbose:
                    print(f"Epoch {e+1}/{self.config.global_epochs}, Client {i}/{len(candidates)}")

                local = c.local_train(server.global_model, e)
                round_directions.append(c.last_update_direction)
                round_weights.append(c.N / N)
                round_losses.append(c.last_loss)
                round_drifts.append(c.last_drift_norm)
                round_states.append(
                    {name: value.detach().clone() for name, value in local.state_dict().items()}
                )

            # Server update
            aggregated_state = self.protocol.aggregate(
                global_state,
                round_states,
                round_weights,
                round_index=e,
            )
            server.model_update(aggregated_state, len(candidates))

            weights = torch.tensor(round_weights, dtype=torch.float32)
            directions = torch.stack(round_directions)
            mean_direction = torch.sum(weights[:, None] * directions, dim=0)
            heterogeneity = torch.sum(
                weights * torch.sum((directions - mean_direction[None, :]) ** 2, dim=1)
            )
            train_loss.append(float(np.average(round_losses, weights=round_weights)))
            update_direction_norm.append(float(torch.linalg.vector_norm(mean_direction)))
            client_drift.append(float(np.average(round_drifts, weights=round_weights)))
            update_direction_dispersion.append(float(heterogeneity))
            selected_clients.append([str(c.client_id) for c in candidates])
            communication_bytes.append(
                self.protocol.round_communication_bytes(len(candidates), model_bytes)
            )

            # Evaluate the model
            index = server.model_eval(clients_set, labtrans)
            train_Cindex.append(index[0])
            train_IBS.append(index[1])

            # Compute test metrics
            model = self._get_model(server.global_model, labtrans)

            pooled_x = np.concatenate(
                [client_data[0] for client_data in clients_set.values()], axis=0
            )
            pooled_y = np.concatenate(
                [client_data[1] for client_data in clients_set.values()], axis=0
            )
            self.adapter.prepare_for_prediction(model, pooled_x, pooled_y, labtrans)
            surv, prediction_diagnostics = self.adapter.predict_survival(
                model,
                data.test_data,
                validation_mode=self.config.prediction_validation,
            )
            ev = EvalSurv(surv, durations_test, events_test, censor_surv="km")
            time_grid = evaluation_time_grid(
                durations_test,
                surv.index.values,
                self.config.evaluation_quantiles,
            )
            test_Cindex.append(ev.concordance_td())
            test_IBS.append(ev.integrated_brier_score(time_grid))

            if self.config.early_stopping:
                # Early stopping
                patience = self.config.early_stopping_patience
                if (
                    len(train_Cindex) > patience
                    and max(train_Cindex[-patience:]) <= train_Cindex[-patience - 1]
                ):
                    break

        self.server_ = server
        self.model_ = model
        self.labtrans_ = labtrans
        self.training_features_ = pooled_x
        self.training_labels_ = pooled_y
        self.last_prediction_diagnostics_ = prediction_diagnostics

        return {
            "train_Cindex": train_Cindex,
            "train_IBS": train_IBS,
            "test_Cindex": test_Cindex,
            "test_IBS": test_IBS,
            "train_loss": train_loss,
            "update_direction_norm": update_direction_norm,
            "client_drift": client_drift,
            "update_direction_dispersion": update_direction_dispersion,
            # Backward-compatible alias. This is not the common-point B^2 in
            # the convergence assumption when local_epochs > 1.
            "empirical_heterogeneity": update_direction_dispersion,
            "selected_clients": selected_clients,
            "communication_bytes": communication_bytes,
            "federated_protocol": self.protocol.name,
            "protocol_metadata": self.protocol.metadata(),
            "telemetry_is_path_average_gradient": (
                self.config.optimizer == "sgd"
                and self.config.weight_decay == 0
                and not self.config.use_differential_privacy
            ),
            "telemetry_is_average_gradient": (
                self.config.optimizer == "sgd"
                and self.config.weight_decay == 0
                and not self.config.use_differential_privacy
            ),
            # With one local SGD step, every recorded direction is evaluated at
            # the broadcast point.  It equals the deterministic local-objective
            # gradient only in full-batch mode; otherwise it is a stochastic
            # gradient at that common point.
            "telemetry_is_common_point_gradient": (
                self.config.optimizer == "sgd"
                and self.config.weight_decay == 0
                and not self.config.use_differential_privacy
                and self.config.local_epochs == 1
            ),
            "telemetry_is_common_point_full_gradient": (
                self.config.optimizer == "sgd"
                and self.config.weight_decay == 0
                and not self.config.use_differential_privacy
                and self.config.local_epochs == 1
                and self.config.full_batch
            ),
        }

    def _run_webdisco_style(self, data, clients_set):
        """Run the protocol-owned distributed linear Cox Newton solver."""
        self.adapter = get_model_adapter(self.config.model_type)
        pooled_x = np.concatenate(
            [client_data[0] for client_data in clients_set.values()], axis=0
        ).astype(np.float32)
        pooled_y = np.concatenate(
            [client_data[1] for client_data in clients_set.values()], axis=0
        ).astype(np.float32)
        durations_train, events_train = self._get_target(pooled_y)
        durations_test, events_test = self._get_target(data.test_label)
        labtrans = self.adapter.configure_targets(self.config, durations_train, events_train)

        server = Server(self.config)
        self.protocol.begin_run(self.config, server.global_model.state_dict())
        fit_result = self.protocol.fit_coefficients(clients_set)

        state = {
            name: value.detach().clone() for name, value in server.global_model.state_dict().items()
        }
        parameter_items = [
            (name, value)
            for name, value in state.items()
            if value.is_floating_point() and value.numel() == self.config.n_features
        ]
        if len(parameter_items) != 1:
            raise RuntimeError(
                "WebDISCO-style requires one bias-free linear CoxPH parameter tensor"
            )
        parameter_name, parameter_value = parameter_items[0]
        state[parameter_name] = torch.as_tensor(
            fit_result.coefficients,
            dtype=parameter_value.dtype,
            device=parameter_value.device,
        ).reshape(parameter_value.shape)
        server.model_update(state)

        model = self._get_model(server.global_model, labtrans)
        self.adapter.prepare_for_prediction(model, pooled_x, pooled_y, labtrans)
        train_survival, train_diagnostics = self.adapter.predict_survival(
            model,
            pooled_x,
            validation_mode=self.config.prediction_validation,
        )
        train_evaluator = EvalSurv(train_survival, durations_train, events_train, censor_surv="km")
        train_grid = evaluation_time_grid(
            durations_train,
            train_survival.index.values,
            self.config.evaluation_quantiles,
        )
        train_cindex = float(train_evaluator.concordance_td())
        train_ibs = float(train_evaluator.integrated_brier_score(train_grid))

        test_survival, prediction_diagnostics = self.adapter.predict_survival(
            model,
            data.test_data,
            validation_mode=self.config.prediction_validation,
        )
        test_evaluator = EvalSurv(test_survival, durations_test, events_test, censor_surv="km")
        test_grid = evaluation_time_grid(
            durations_test,
            test_survival.index.values,
            self.config.evaluation_quantiles,
        )
        test_cindex = float(test_evaluator.concordance_td())
        test_ibs = float(test_evaluator.integrated_brier_score(test_grid))

        self.server_ = server
        self.model_ = model
        self.labtrans_ = labtrans
        self.training_features_ = pooled_x
        self.training_labels_ = pooled_y
        self.last_prediction_diagnostics_ = prediction_diagnostics
        self.webdisco_fit_result_ = fit_result
        metadata = self.protocol.metadata()
        metadata.update(
            {
                "converged": bool(fit_result.converged),
                "n_iterations": int(fit_result.n_iterations),
                "score_norm": float(fit_result.score_norm),
                "payload_bytes_estimate": int(fit_result.communication_bytes),
                "training_prediction_diagnostics": {
                    "valid": bool(train_diagnostics.valid),
                    "identical_across_samples": bool(train_diagnostics.identical_across_samples),
                },
            }
        )
        return {
            "train_Cindex": [train_cindex],
            "train_IBS": [train_ibs],
            "test_Cindex": [test_cindex],
            "test_IBS": [test_ibs],
            "train_loss": [-float(fit_result.log_likelihood_history[-1])],
            "update_direction_norm": [float(np.linalg.norm(fit_result.coefficients))],
            "client_drift": [float("nan")],
            "update_direction_dispersion": [float("nan")],
            "empirical_heterogeneity": [float("nan")],
            "selected_clients": [[str(key) for key in clients_set]],
            "communication_bytes": [int(fit_result.communication_bytes)],
            "federated_protocol": self.protocol.name,
            "protocol_metadata": metadata,
            "telemetry_is_path_average_gradient": False,
            "telemetry_is_average_gradient": False,
            "telemetry_is_common_point_gradient": False,
            "telemetry_is_common_point_full_gradient": False,
        }

    def predict_survival(self, features: np.ndarray) -> pd.DataFrame:
        """Predict survival curves after :meth:`run` has fitted a model."""
        if not hasattr(self, "model_"):
            raise RuntimeError("run must be called before predict_survival")
        survival, diagnostics = self.adapter.predict_survival(
            self.model_,
            features,
            validation_mode=self.config.prediction_validation,
        )
        self.last_prediction_diagnostics_ = diagnostics
        return survival

    def get_privacy_info(self) -> Dict[str, Any]:
        """
        Get differential privacy protection information.

        Returns:
            dict: Privacy protection related information.
        """
        if not self.config.use_differential_privacy:
            return {"privacy_protection": False}

        from .differential_privacy import DifferentialPrivacy

        dp_tool = DifferentialPrivacy(self.config)

        # Compute privacy budget consumption
        total_epsilon, per_round_epsilon = dp_tool.compute_privacy_budget(
            self.config.global_epochs, self.config.num_clients
        )

        mechanism = self.config.dp_mechanism if hasattr(self.config, "dp_mechanism") else "gaussian"

        privacy_info = {
            "privacy_protection": True,
            "mechanism": mechanism,
            "epsilon": self.config.dp_epsilon,
            "sensitivity": self.config.dp_sensitivity,
            "total_epsilon": total_epsilon,
            "per_round_epsilon": per_round_epsilon,
            "formal_accounting_available": False,
            "privacy_scope": "experimental clipped client-update perturbation",
        }

        # Add Gaussian mechanism specific parameters
        if mechanism == "gaussian":
            privacy_info.update(
                {
                    "delta": self.config.dp_delta,
                    "noise_multiplier": self.config.dp_noise_multiplier,
                    "noise_scale": dp_tool.get_noise_scale(self.config.num_clients),
                    # Per-element standard deviation actually added to the
                    # clipped client update; compare it with typical weight
                    # magnitudes to tell whether the noise swamps the signal.
                    "effective_noise_sigma": (
                        dp_tool.get_noise_scale(self.config.num_clients)
                        * self.config.dp_clip_norm
                    ),
                }
            )

        # Add gradient clipping parameters (Gaussian and Laplace mechanisms)
        if mechanism in ["gaussian", "laplace"]:
            privacy_info["clip_norm"] = self.config.dp_clip_norm

        return privacy_info

    def _get_target(self, df):
        """Get the target variables."""
        return df[:, 0], df[:, 1]

    def plot_results(
        self,
        results: Dict[str, List[float]],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Figure:
        """
        Plot training results.

        Args:
            results: Training results in the form:
                {
                    'train_Cindex': List[float],  # Training set C-index
                    'train_IBS': List[float],     # Training set IBS
                    'test_Cindex': List[float],   # Test set C-index
                    'test_IBS': List[float]       # Test set IBS
                }
            output_path: Optional path to save the figure.
            show: Whether to open an interactive window; set to False for automated tests.

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        # Use a more modern style and avoid the deprecated seaborn style
        try:
            # Try the seaborn-v0_8 style (matplotlib 3.6+)
            plt.style.use("seaborn-v0_8")
        except OSError:
            # Fall back to the default style and set parameters manually
            plt.style.use("default")

        # Set fonts and style
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        # Close all existing figure windows to avoid multiple windows
        plt.close("all")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Get the number of epochs and convert to a list
        n_epochs = len(results["train_Cindex"])
        epochs = list(range(1, n_epochs + 1))

        # Set the C-index axis range
        ax1.set_xlim(1, n_epochs)
        cindex_min = min(min(results["train_Cindex"]), min(results["test_Cindex"]))
        cindex_max = max(max(results["train_Cindex"]), max(results["test_Cindex"]))
        ax1.set_ylim(cindex_min - 0.05, cindex_max + 0.05)

        # Plot the C-index
        ax1.plot(
            epochs,
            results["train_Cindex"],
            color="#2ecc71",
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            markevery=5,
        )
        ax1.plot(
            epochs,
            results["test_Cindex"],
            color="#3498db",
            label="Testing",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=4,
            markevery=5,
        )

        ax1.set_title("Concordance Index")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("C-index")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

        # Add final value annotations
        final_train_cindex = results["train_Cindex"][-1]
        final_test_cindex = results["test_Cindex"][-1]
        ax1.text(
            0.02,
            0.98,
            f"Final Train C-index: {final_train_cindex:.3f}\nFinal Test C-index: {final_test_cindex:.3f}",
            transform=ax1.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # Set the IBS axis range
        ax2.set_xlim(1, n_epochs)
        ibs_min = min(min(results["train_IBS"]), min(results["test_IBS"]))
        ibs_max = max(max(results["train_IBS"]), max(results["test_IBS"]))
        ax2.set_ylim(max(0, ibs_min - 0.05), min(1.0, ibs_max + 0.05))

        # Plot the IBS
        ax2.plot(
            epochs,
            results["train_IBS"],
            color="#2ecc71",
            label="Training",
            linewidth=2,
            marker="o",
            markersize=4,
            markevery=5,
        )
        ax2.plot(
            epochs,
            results["test_IBS"],
            color="#3498db",
            label="Testing",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=4,
            markevery=5,
        )

        ax2.set_title("Integrated Brier Score")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IBS")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

        # Add final value annotations
        final_train_ibs = results["train_IBS"][-1]
        final_test_ibs = results["test_IBS"][-1]
        ax2.text(
            0.02,
            0.98,
            f"Final Train IBS: {final_train_ibs:.3f}\nFinal Test IBS: {final_test_ibs:.3f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        # Adjust layout
        plt.tight_layout()

        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig
