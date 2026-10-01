# -*- coding: UTF-8 -*-
"""
Example 06: Data Augmentation (MVAEC / MVAES)

Demonstrates two VAE-based federated data augmentation methods:
  - MVAEC: each client trains a VAE on local data and generates augmented samples (stronger privacy)
  - MVAES: the server collects clients' augmented data and redistributes it (better diversity)

Note: augmentation requires each client to have at least 10 samples and at least 1 uncensored sample.

API path: FSARunner.run(data, type='raw_aug', aug_method='MVAEC'/'MVAES')
Run: python examples/06_data_augmentation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.config import FSAConfig
from federated_survival.core.runner import FSARunner


def main():
    print("=== Example 06: Data Augmentation ===\n")

    # Prepare data (slightly larger sample size to ensure each client
    # has enough uncensored samples for VAE training). A Dirichlet split makes
    # the client label/time distributions heterogeneous, which is where
    # augmentation actually helps: on homogeneous IID splits plain federated
    # training is already near the centralized optimum and augmentation adds
    # noise instead of signal.
    gen = DataGenerator(SimulationConfig(n_samples=400, n_features=10, random_state=42))
    data = gen.generate("weibull", c_mean=0.4)
    splitter = DataSplitter(
        n_clients=3, split_type="dirichlet", alpha=0.5,
        test_size=0.2, random_state=42)
    dataset = splitter.split(data)

    # Shared config (including augmentation parameters)
    # Values follow the tuning recipe validated in
    # docs/augmentation-privacy.md ("Tuned augmentation configurations"):
    #   - latent_num at/below the feature count is the single most impactful
    #     change (an over-sized latent space yields noisy synthetic rows);
    #   - beta=5 keeps synthetic event times realistic;
    #   - k=0.5 with sparse latent sampling balances augmentation volume vs. noise;
    #   - measured on this dataset: raw 0.545 < MVAEC 0.600 < MVAES 0.610.
    base = dict(
        num_clients=3, n_features=10, n_samples=400,
        model_type="DeepSurv",
        num_nodes=(32, 32), dropout=0.1,
        local_epochs=1, global_epochs=30,
        learning_rate=0.003, batch_size=32,
        weight_decay=0.0, early_stopping=False,
        random_seed=42, verbose=False,
        # Data augmentation (VAE) parameters
        latent_num=10,    # Latent space dimension (keep at/below n_features)
        hidden_num=32,    # Hidden layer dimension
        alpha=1.0,        # KL divergence weight
        beta=5.0,         # Conditional loss weight
        k=0.5,            # Augmentation ratio (generated samples = k x original)
        augmentation_sampling="sparse",     # Sparse sampling of latent codes
        augmentation_sparse_gamma=0.1,      # Sparsity strength
    )

    # 1) Without augmentation
    print("1) Without augmentation (raw)")
    r1 = FSARunner(FSAConfig(**base)).run(dataset, type="raw")
    print("   Test C-index={:.4f}, IBS={:.4f}".format(
        r1["test_Cindex"][-1], r1["test_IBS"][-1]))

    # 2) MVAEC client-side local augmentation
    print("\n2) MVAEC client-side local augmentation")
    r2 = FSARunner(FSAConfig(**base)).run(dataset, type="raw_aug", aug_method="MVAEC")
    print("   Test C-index={:.4f}, IBS={:.4f}".format(
        r2["test_Cindex"][-1], r2["test_IBS"][-1]))

    # 3) MVAES server-side centralized augmentation and redistribution
    print("\n3) MVAES server-side centralized augmentation")
    r3 = FSARunner(FSAConfig(**base)).run(dataset, type="raw_aug", aug_method="MVAES")
    print("   Test C-index={:.4f}, IBS={:.4f}".format(
        r3["test_Cindex"][-1], r3["test_IBS"][-1]))

    print("\nSelection tips:")
    print("  MVAEC - Privacy first; client data never leaves the client; low communication overhead")
    print("  MVAES - Diversity first; server redistributes; suits clients with small data volumes")
    print("  Tuning: k controls the augmentation amount; latent_num/hidden_num control VAE capacity")
    print("  Data: augmentation pays off on heterogeneous splits (Dirichlet/censoring);")
    print("        on homogeneous IID splits it usually adds noise instead of signal")
    print("\n=== Example 06 Done ===")


if __name__ == "__main__":
    main()
