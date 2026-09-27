"""Loss adapters used to keep code-level objectives explicit."""

from __future__ import annotations

import torch


class ScaledLoss(torch.nn.Module):
    """Multiply a pycox loss by a fixed positive scale."""

    def __init__(self, base_loss: torch.nn.Module, scale: float):
        super().__init__()
        if not 0 < scale <= 1:
            raise ValueError("loss scale must be in (0, 1]")
        self.base_loss = base_loss
        self.scale = float(scale)

    def forward(self, *args, **kwargs):
        return self.scale * self.base_loss(*args, **kwargs)


def apply_cox_patient_normalization(model, config, event_fraction: float) -> None:
    """Convert pycox's event-mean Cox loss to a patient-mean loss.

    Pycox divides Cox losses by the number of observed events. The
    patient-normalized option instead divides the sum of event contributions
    by the number of patients. Multiplication by m_k / n_k makes these
    definitions identical.
    """
    cox_models = {"CoxPH", "DeepSurv", "CoxCC", "CoxTime"}
    if config.model_type in cox_models and config.cox_loss_normalization == "patient":
        model.loss = ScaledLoss(model.loss, event_fraction)
