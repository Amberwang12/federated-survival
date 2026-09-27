"""Latent sampling, reconstruction, and the conditional-MSE objective.

``MVAE`` backs the MVAEC / MVAES augmentation paths.  ``sample`` and
``generate`` are the entry points used to synthesise new client rows, and
``cmse`` is the conditional mean-squared-error term in the training objective.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from federated_survival.core.mvae import Decoder, DecoderTime, Encoder, MVAE, cmse


def _tiny_mvae(feature_dim=3, latent_dim=2, hidden=4):
    """A correctly-wired MVAE: the decoder consumes ``latent_dim`` inputs."""
    return MVAE(
        Encoder(feature_dim, hidden, hidden),
        Decoder(latent_dim, hidden, feature_dim),
        DecoderTime(latent_dim, hidden, 1),
        latent_dim=latent_dim,
        encoder_out=hidden,
    )


def test_sample_returns_the_requested_number_of_rows():
    features, targets = _tiny_mvae().sample(7)

    assert features.shape == (7, 3)
    assert targets.shape == (7, 2)
    assert np.isfinite(features).all()
    assert np.allclose(targets[:, 1], 1.0)  # 采样样本的 status 固定为 1


def test_sample_handles_zero_samples():
    features, targets = _tiny_mvae().sample(0)

    assert features.shape == (0, 3)
    assert targets.shape == (0, 2)


def test_generate_reconstructs_a_batch():
    mvae = _tiny_mvae()
    batch = torch.randn(5, 3)

    reconstructed = mvae.generate(batch.numpy())

    assert isinstance(reconstructed, tuple) and len(reconstructed) == 2
    assert reconstructed[0].shape == (5, 3)
    assert reconstructed[1].shape == (5, 1)


def test_cmse_counts_events_and_early_censored_patients():
    pre = torch.tensor([[1.0], [5.0], [3.0]])
    time = torch.tensor([[2.0], [4.0], [3.0]])
    status = torch.tensor([[0.0], [0.0], [1.0]])

    # 第 0 行：删失且 pre < time  -> 计入，误差 (1-2)^2
    # 第 1 行：删失且 pre >= time -> 排除
    # 第 2 行：事件              -> 计入，误差 0
    expected = ((1.0 - 2.0) ** 2 + 0.0 + 0.0) / 3.0

    assert cmse(pre, time, status).item() == pytest.approx(expected)


def test_cmse_is_zero_when_every_row_is_excluded():
    pre = torch.tensor([[9.0], [9.0]])
    time = torch.tensor([[1.0], [1.0]])
    status = torch.tensor([[0.0], [0.0]])

    assert cmse(pre, time, status).item() == 0.0
