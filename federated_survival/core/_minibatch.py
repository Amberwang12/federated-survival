"""Exact-step mini-batch utilities for the Local-SGD implementation.

PyCox normally interprets ``epochs`` as complete passes through a shuffled
dataset. The package uses ``E`` for the number of stochastic local updates
between two server aggregations. The helpers below make that mapping literal:
one call performs exactly ``steps`` independently sampled batches.
"""

from __future__ import annotations

import hashlib
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
import torchtuples as tt


def deterministic_stream_seed(base_seed: int, namespace: str, round_index: int = 0) -> int:
    """Return a stable PyTorch seed without relying on Python's salted hash."""
    payload = f"{int(base_seed)}:{namespace}:{int(round_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class IndependentBatchSampler:
    """Sample independent fixed-size subsets for a prescribed number of steps.

    Sampling is without replacement *inside* a batch and independent across
    batches.  For CoxPH/DeepSurv, ``event_indices`` conditions the sampler on
    every batch containing at least one observed event, which avoids the
    undefined zero-event Cox denominator while retaining a well-defined batch
    distribution.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        steps: int,
        seed: int,
        event_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0 or steps <= 0:
            raise ValueError("dataset_size, batch_size and steps must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = min(int(batch_size), self.dataset_size)
        self.steps = int(steps)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self.event_indices = None
        if event_indices is not None:
            values = {int(value) for value in event_indices}
            if not values:
                raise ValueError("event-aware mini-batches require at least one event")
            if min(values) < 0 or max(values) >= self.dataset_size:
                raise ValueError("event index is outside the dataset")
            self.event_indices = values

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.steps):
            while True:
                indices = torch.randperm(
                    self.dataset_size, generator=self.generator
                )[: self.batch_size].tolist()
                if self.event_indices is None or any(
                    index in self.event_indices for index in indices
                ):
                    yield indices
                    break

    def __len__(self) -> int:
        return self.steps


def make_independent_step_dataloader(
    model,
    inputs,
    targets,
    batch_size: int,
    steps: int,
    model_type: str,
    seed: int,
):
    """Build the model-specific PyCox dataset with an independent batch sampler."""
    template = model.make_dataloader(
        (inputs, targets), batch_size=min(batch_size, len(inputs)), shuffle=False, num_workers=0
    )
    event_indices = None
    if model_type in {"CoxPH", "DeepSurv"}:
        events = np.asarray(targets[1]).reshape(-1)
        event_indices = np.flatnonzero(events > 0).tolist()
    sampler = IndependentBatchSampler(
        dataset_size=len(template.dataset),
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        event_indices=event_indices,
    )
    return tt.data.DataLoaderBatch(
        template.dataset,
        batch_sampler=sampler,
        num_workers=0,
    )


def fit_in_local_steps(
    model,
    inputs,
    targets,
    *,
    model_type: str,
    batch_size: int,
    steps: int,
    full_batch: bool,
    seed: int,
):
    """Fit a PyCox model for exactly ``steps`` optimizer updates."""
    if full_batch:
        kwargs = {
            "batch_size": len(inputs),
            "epochs": steps,
            "verbose": False,
        }
        if model_type == "PC-Hazard":
            kwargs["check_out_features"] = False
        return model.fit(inputs, targets, **kwargs)

    # Cox models use this attribute later for Breslow baseline-hazard
    # estimation. Their regular ``fit`` method sets it, while a direct
    # ``fit_dataloader`` call does not.
    if model_type in {"CoxPH", "DeepSurv", "CoxCC", "CoxTime"}:
        stored_inputs, stored_targets = inputs, targets
        if model_type in {"CoxCC", "CoxTime"}:
            stored_inputs, stored_targets = model._sorted_input_target(inputs, targets)
        model.training_data = tt.tuplefy(stored_inputs, stored_targets)

    dataloader = make_independent_step_dataloader(
        model,
        inputs,
        targets,
        batch_size=min(batch_size, len(inputs)),
        steps=steps,
        model_type=model_type,
        seed=seed,
    )
    kwargs = {"epochs": 1, "verbose": False}
    if model_type == "PC-Hazard":
        kwargs["check_out_features"] = False
    return model.fit_dataloader(dataloader, **kwargs)
