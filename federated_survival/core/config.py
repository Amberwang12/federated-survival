from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import torch.nn as nn


@dataclass
class FSAConfig:
    """Configuration class for federated survival analysis."""

    # Data parameters
    dataset_name: Optional[str] = None  # Dataset name when using real data
    mode: str = "simulate"  # Options: 'real', 'simulate'

    # Data configuration
    n_samples: int = 1000  # Number of samples
    n_features: int = 20  # Number of features
    censor_rate: float = 0.4  # Censoring rate
    data_type: str = "weibull"  # Options: 'weibull', 'lognormal', '1', '2', '3', '4'
    sim_type: str = "1"  # Simulation data type

    # Model parameters
    model_type: str = "PC-Hazard"  # Model type
    num_nodes: Tuple[int, ...] = (32, 32)  # Number of layers and nodes per layer
    num_durations: int = 25  # Number of discretized time intervals
    batch_norm: bool = False  # Whether to use batch normalization
    dropout: float = 0.1  # Dropout rate
    activation: nn.Module = nn.ReLU  # Activation function

    # Federated learning parameters
    num_clients: int = 5  # Number of clients
    global_epochs: int = 50  # Number of global training rounds
    early_stopping: bool = False  # Whether to use early stopping
    early_stopping_patience: int = 10  # Early stopping patience (rounds)
    local_epochs: int = 1  # Local update steps E before each aggregation (legacy field name)
    batch_size: int = 32  # Batch size
    full_batch: bool = False  # False for independent mini-batch local SGD; True only as a full-batch special case
    learning_rate: float = 1e-3  # Learning rate
    optimizer: str = "adam"  # Options: 'adam', 'sgd'
    weight_decay: float = 0.05  # Optimizer weight decay
    client_sample_ratio: float = 1.0  # Fraction of clients selected per round
    split_method: str = "iid"  # Data split method, options: 'iid', 'non-iid', 'time-non-iid'

    # Federated protocol parameters
    federated_protocol: str = "FedAvg"  # FedAvg, FedProx, FedOpt, WebDISCO-style
    proximal_mu: float = 0.01  # FedProx proximal term coefficient
    server_optimizer: str = "adam"  # FedOpt: adam, yogi, adagrad
    server_learning_rate: float = 0.01
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 1e-3
    webdisco_max_iter: int = 50
    webdisco_tolerance: float = 1e-7
    webdisco_l2: float = 0.0
    webdisco_ridge: float = 1e-8
    webdisco_max_step_norm: float = 5.0

    # Data augmentation parameters
    k: float = 0.5  # Proportion of augmented data
    latent_num: int = 10
    hidden_num: int = 30
    alpha: float = 1.0
    beta: float = 1.0
    augmentation_sampling: str = "sparse"
    augmentation_sparse_gamma: float = 0.1

    # Differential privacy parameters
    use_differential_privacy: bool = False  # Whether to use differential privacy
    dp_mechanism: str = "gaussian"  # Differential privacy mechanism: 'gaussian', 'laplace', 'exponential'
    dp_epsilon: float = 1.0  # Privacy budget (epsilon)
    dp_delta: float = 1e-5  # Failure probability (delta) - required only for the Gaussian mechanism
    dp_sensitivity: float = 1.0  # Sensitivity
    dp_noise_multiplier: float = 1.0  # Noise multiplier - used only by the Gaussian mechanism
    dp_clip_norm: float = 1.0  # Gradient clipping norm

    # Other parameters
    verbose: bool = False  # Whether to print verbose output
    show_progress: bool = False  # Whether to show the tqdm progress bar
    random_seed: int = 42  # Random seed
    evaluation_quantiles: Tuple[float, float] = (0.05, 0.95)
    cox_loss_normalization: str = "patient"  # 'event' reproduces raw pycox scaling
    prediction_validation: str = "raise"  # 'raise', 'warn', or 'none'

    # Base configuration
    n_rounds: int = 100
    random_state: Optional[int] = None

    # Model configuration
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Data split configuration
    split_alpha: float = 0.5  # Dirichlet distribution parameter for non-iid splits
    test_size: float = 0.2

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate mode
        valid_modes = ["real", "simulate"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")

        # A dataset name is required in 'real' mode
        if self.mode == "real" and self.dataset_name is None:
            raise ValueError("dataset_name must be provided when mode is 'real'")

        if not 0 <= self.censor_rate <= 1:
            raise ValueError("censor_rate must be between 0 and 1")

        if not 0 < self.client_sample_ratio <= 1:
            raise ValueError("client_sample_ratio must be between 0 and 1")

        self.optimizer = self.optimizer.lower()
        if self.optimizer not in ["adam", "sgd"]:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.global_epochs <= 0 or self.local_epochs <= 0:
            raise ValueError("global_epochs and local_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        from ..protocols import get_federated_protocol

        protocol = get_federated_protocol(self.federated_protocol)
        self.federated_protocol = protocol.name
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        self.server_optimizer = self.server_optimizer.lower()
        if self.server_optimizer not in ["adam", "yogi", "adagrad"]:
            raise ValueError("server_optimizer must be 'adam', 'yogi', or 'adagrad'")
        if self.server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be positive")
        if not 0 <= self.server_beta1 < 1 or not 0 <= self.server_beta2 < 1:
            raise ValueError("server_beta1 and server_beta2 must be in [0, 1)")
        if self.server_tau <= 0:
            raise ValueError("server_tau must be positive")
        if self.webdisco_max_iter <= 0:
            raise ValueError("webdisco_max_iter must be positive")
        if self.webdisco_tolerance <= 0:
            raise ValueError("webdisco_tolerance must be positive")
        if self.webdisco_l2 < 0 or self.webdisco_ridge < 0:
            raise ValueError("webdisco_l2 and webdisco_ridge must be non-negative")
        if self.webdisco_max_step_norm <= 0:
            raise ValueError("webdisco_max_step_norm must be positive")
        self.cox_loss_normalization = self.cox_loss_normalization.lower()
        if self.cox_loss_normalization not in ["patient", "event"]:
            raise ValueError("cox_loss_normalization must be 'patient' or 'event'")
        self.prediction_validation = self.prediction_validation.lower()
        if self.prediction_validation not in ["raise", "warn", "none"]:
            raise ValueError("prediction_validation must be 'raise', 'warn', or 'none'")
        q_low, q_high = self.evaluation_quantiles
        if not 0 <= q_low < q_high <= 1:
            raise ValueError("evaluation_quantiles must satisfy 0 <= low < high <= 1")

        # Validate model type
        # Resolve model support through the public registry so third-party
        # adapters can be configured without editing this dataclass.
        from ..models import available_model_adapters

        valid_model_types = list(available_model_adapters())
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}")

        # Validate intermediate layer parameters for CoxPH
        if self.model_type == "CoxPH":
            if self.num_nodes != ():
                raise ValueError("num_nodes must be () for CoxPH")

        protocol.validate_config(self)

        # Validate data split method
        self.split_method = self.split_method.lower()
        valid_split_methods = [
            "iid",
            "random",
            "non-iid",
            "censoring-non-iid",
            "time-non-iid",
            "dirichlet",
        ]
        if self.split_method not in valid_split_methods:
            raise ValueError(f"split_method must be one of {valid_split_methods}")

        # Validate numerical ranges
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.k < 1:
            raise ValueError("k must be between 0 and 1")

        # Validate data augmentation parameters
        if self.latent_num <= 0:
            raise ValueError("latent_num must be positive")
        if self.hidden_num <= 0:
            raise ValueError("hidden_num must be positive")
        self.augmentation_sampling = self.augmentation_sampling.lower().replace("-", "_")
        if self.augmentation_sampling not in ["sparse", "unconditional"]:
            raise ValueError("augmentation_sampling must be 'sparse' or 'unconditional'")
        if self.augmentation_sparse_gamma < 0:
            raise ValueError("augmentation_sparse_gamma must be non-negative")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        # Validate num_durations
        if self.num_durations <= 0:
            raise ValueError("num_durations must be positive")

        # Validate dropout
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

        # Validate n_rounds
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive")

        # Validate split_alpha
        if self.split_alpha <= 0:
            raise ValueError("split_alpha must be positive")

        # Validate differential privacy parameters
        if self.use_differential_privacy:
            # Validate mechanism type
            valid_dp_mechanisms = ["gaussian", "laplace", "exponential"]
            if self.dp_mechanism not in valid_dp_mechanisms:
                raise ValueError(f"dp_mechanism must be one of {valid_dp_mechanisms}")

            # Validate common parameters
            if self.dp_epsilon <= 0:
                raise ValueError("dp_epsilon must be positive")
            if self.dp_sensitivity <= 0:
                raise ValueError("dp_sensitivity must be positive")

            # Gaussian mechanism specific parameters
            if self.dp_mechanism == "gaussian":
                if not 0 < self.dp_delta < 1:
                    raise ValueError("dp_delta must be between 0 and 1 for Gaussian mechanism")
                if self.dp_noise_multiplier <= 0:
                    raise ValueError("dp_noise_multiplier must be positive for Gaussian mechanism")

            # Validate gradient clipping parameters (applies to Gaussian and Laplace mechanisms)
            if self.dp_mechanism in ["gaussian", "laplace"]:
                if self.dp_clip_norm <= 0:
                    raise ValueError("dp_clip_norm must be positive")

            # Warn when a configured DP field cannot influence the selected
            # mechanism.  The Gaussian noise scale is driven by
            # dp_noise_multiplier and the Laplace scale by dp_epsilon, so
            # switching dp_mechanism turns one of them into an inert value that
            # looks like a working privacy control.  dp_delta never affects the
            # noise scale either.  See core.differential_privacy for the
            # authoritative mapping.
            from .differential_privacy import DP_MECHANISM_INERT_KNOBS

            inert = DP_MECHANISM_INERT_KNOBS[self.dp_mechanism]
            driver = (
                "dp_noise_multiplier" if self.dp_mechanism == "gaussian" else "dp_epsilon"
            )
            self._warn_inactive_dp_parameters(inert, driver)

        # Set default model parameters
        if not self.model_params:
            if self.model_type == "PC-Hazard":
                self.model_params = {"n_intervals": 10, "hidden_size": 32, "dropout": 0.1}
            elif self.model_type == "LogisticHazard":
                self.model_params = {"n_intervals": 10, "hidden_size": 32, "dropout": 0.1}
            elif self.model_type == "DeepHit":
                self.model_params = {"n_intervals": 10, "hidden_size": 32, "dropout": 0.1}
            elif self.model_type == "CoxTime":
                self.model_params = {"hidden_size": 32, "dropout": 0.1}
            elif self.model_type in ["DeepSurv", "CoxPH"]:
                self.model_params = {"l2_reg": 0.01}
            elif self.model_type == "CoxCC":
                self.model_params = {"l2_reg": 0.01}

    def _warn_inactive_dp_parameters(self, inert, driver) -> None:
        """Emit one warning per inert DP field that was set away from its default.

        A field left at its default cannot mislead anyone, so it is silent.  The
        warning fires only when the user has actually chosen a non-default value
        for a parameter that the selected mechanism ignores.
        """
        import warnings
        from dataclasses import MISSING, fields as dataclass_fields

        defaults = {}
        for f in dataclass_fields(self):
            if f.default is not MISSING:
                defaults[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                defaults[f.name] = f.default_factory()  # type: ignore[misc]

        for name in inert:
            value = getattr(self, name, None)
            default = defaults.get(name)
            if default is not None and value == default:
                # Untouched default: not an active misconfiguration.
                continue
            warnings.warn(
                f"dp_mechanism='{self.dp_mechanism}' does not use {name} "
                f"(currently {value!r}); the injected noise is controlled by "
                f"{driver}. Setting {name} has no effect on the noise added to "
                f"model updates. Switch dp_mechanism or adjust {driver} instead.",
                UserWarning,
                stacklevel=4,
            )
