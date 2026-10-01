"""
Server class
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from pycox.evaluation import EvalSurv
import numpy as np
from .differential_privacy import DifferentialPrivacy
from ..utils.metrics import evaluation_time_grid
from ..models import get_model_adapter

class Server:
    """Server class."""
    
    def __init__(self, config):
        """
        Initialize the server.

        Args:
            config: Federated learning configuration.
        """
        self.config = config
        self.adapter = get_model_adapter(config.model_type)
        self.global_model = self._create_model()
        
        # Initialize the differential privacy tool
        if self.config.use_differential_privacy:
            self.dp_tool = DifferentialPrivacy(config)
        else:
            self.dp_tool = None
        
    def _create_model(self) -> nn.Module:
        """Create the global model."""
        return self.adapter.build_network(self.config)
        
    def model_update(self, weight_accumulator: Dict[str, torch.Tensor], num_clients: int = 1):
        """
        Update the global model.

        Args:
            weight_accumulator: The weight accumulator.
            num_clients: Number of clients participating in the aggregation.
        """
        # Note: differential privacy noise is already added during local client
        # training; no additional noise is applied here.
        # FedAvg: replace the global model parameters with the weighted average directly.
        for name, data in self.global_model.state_dict().items():
            data.copy_(weight_accumulator[name].to(dtype=data.dtype, device=data.device))
            
    def model_eval(self, client_set, labtrans) -> tuple:
        """
        Evaluate the model.
        Compute the C-index and IBS for each client and return their averages.

        Args:
            client_set: Client data.
            labtrans: Label transformer.

        Returns:
            tuple: (C-index, IBS)
        """
        model = self._get_model(self.global_model, labtrans)
        pooled_x = np.concatenate([value[0] for value in client_set.values()], axis=0)
        pooled_y = np.concatenate([value[1] for value in client_set.values()], axis=0)
        self.adapter.prepare_for_prediction(model, pooled_x, pooled_y, labtrans)
        surv, self.last_prediction_diagnostics = self.adapter.predict_survival(
            model,
            pooled_x,
            validation_mode=self.config.prediction_validation,
        )
        durations_train, events_train = self._get_target(pooled_y)
        ev = EvalSurv(surv, durations_train, events_train, censor_surv='km')
        time_grid = evaluation_time_grid(
            durations_train,
            surv.index.values,
            self.config.evaluation_quantiles,
        )
        return float(ev.concordance_td()), float(ev.integrated_brier_score(time_grid))
        
    def _get_model(self, net: nn.Module, labtrans=None):
        """Build the survival analysis model."""
        return self.adapter.build_model(
            net,
            self.config,
            label_transform=labtrans,
            optimizer=torch.optim.Adam,
        )
        
    def _get_target(self, df):
        """Extract the target variables."""
        return df[:, 0], df[:, 1] 
