"""Train-only preprocessing shared by YAML and composable experiments."""

from typing import Dict, Tuple

import numpy as np

from .splitter import DataSet


def preprocess_partition(
    dataset: DataSet, missing_values: str = "error", standardize: bool = False
) -> DataSet:
    """Impute from training medians and/or standardize using training statistics."""
    if missing_values not in ("error", "drop", "median"):
        raise ValueError("missing_values must be error, drop, or median")
    train = dataset.train_data.astype(float).copy()
    test = dataset.test_data.astype(float).copy()
    clients: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        key: (features.astype(float).copy(), labels)
        for key, (features, labels) in dataset.clients_set.items()
    }
    if missing_values == "median":
        medians = np.nanmedian(train, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("a feature is entirely missing in the training set")

        def impute(array):
            rows, columns = np.where(np.isnan(array))
            array[rows, columns] = medians[columns]
            return array

        train = impute(train)
        test = impute(test)
        clients = {key: (impute(features), labels) for key, (features, labels) in clients.items()}
    arrays = [train, test] + [features for features, _ in clients.values()]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("feature columns contain NaN or infinity after preprocessing")
    if standardize:
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale < 1e-8] = 1.0
        train = (train - mean) / scale
        test = (test - mean) / scale
        clients = {
            key: ((features - mean) / scale, labels) for key, (features, labels) in clients.items()
        }
    return DataSet(
        clients_set={
            key: (features.astype("float32"), labels.astype("float32"))
            for key, (features, labels) in clients.items()
        },
        train_data=train.astype("float32"),
        train_label=dataset.train_label,
        test_data=test.astype("float32"),
        test_label=dataset.test_label,
        raw_aug_clients_set=dataset.raw_aug_clients_set,
    )
