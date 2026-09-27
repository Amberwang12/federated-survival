import torch

from federated_survival.core._losses import ScaledLoss
from federated_survival.core._minibatch import IndependentBatchSampler


def test_scaled_loss_multiplies_base_value():
    base = torch.nn.MSELoss()
    scaled = ScaledLoss(base, 0.25)
    prediction = torch.tensor([2.0])
    target = torch.tensor([0.0])
    assert torch.isclose(scaled(prediction, target), torch.tensor(1.0))


def test_independent_sampler_has_exact_step_count_and_batch_size():
    sampler = IndependentBatchSampler(20, batch_size=6, steps=7, seed=42)
    batches = list(sampler)
    assert len(batches) == 7
    assert all(len(batch) == 6 for batch in batches)
    assert all(len(set(batch)) == 6 for batch in batches)


def test_event_aware_sampler_never_emits_zero_event_cox_batch():
    event_indices = [1, 9]
    sampler = IndependentBatchSampler(
        12, batch_size=4, steps=50, seed=7, event_indices=event_indices
    )
    assert all(set(batch).intersection(event_indices) for batch in sampler)


def test_independent_sampler_is_reproducible():
    left = list(IndependentBatchSampler(15, 5, 4, seed=11))
    right = list(IndependentBatchSampler(15, 5, 4, seed=11))
    assert left == right
