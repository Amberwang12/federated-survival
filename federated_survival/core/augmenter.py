import numpy as np
import torch
from typing import Dict, Tuple, Optional
from .mvae import vae_train
import random


class DataAugmenter:
    """Data augmenter supporting the MVAES and MVAEC augmentation methods."""

    def __init__(
        self,
        latent_num: int = 10,
        hidden_num: int = 30,
        alpha: float = 1.0,
        beta: float = 1.0,
        sampling: str = "sparse",
        sparse_gamma: float = 0.1,
    ):
        """
        Initialize the data augmenter.

        Args:
            latent_num: Dimension of the latent space.
            hidden_num: Dimension of the hidden layers.
            alpha: Weight of the KL divergence loss.
            beta: Weight of the conditional loss.
            sampling: ``sparse`` performs conditional sampling around the
                latent codes of the time-sparse half region;
                ``unconditional`` samples from the standard normal latent space.
            sparse_gamma: Standard deviation of the noise added around the
                latent codes of the sparse region.
        """
        sampling = str(sampling).lower().replace("-", "_")
        if sampling not in {"sparse", "unconditional"}:
            raise ValueError("sampling must be 'sparse' or 'unconditional'")
        if sparse_gamma < 0:
            raise ValueError("sparse_gamma must be non-negative")
        self.latent_num = latent_num
        self.hidden_num = hidden_num
        self.alpha = alpha
        self.beta = beta
        self.sampling = sampling
        self.sparse_gamma = float(sparse_gamma)
        self.sampling_regions = {}
        self.last_sampling_region = None

    @staticmethod
    def _sparse_region(times: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Return indices in the wider median half of observed event times.

        This follows the original FSA-MVAE implementation: split uncensored
        times at their median and call the half with the wider time span the
        sparse region.  A tie is resolved in favour of the later-time half.
        """
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("sparse sampling requires at least one uncensored sample")
        if not np.all(np.isfinite(times)):
            raise ValueError("uncensored event times must be finite")

        median = float(np.median(times))
        # Preserve the legacy boundary convention: median observations belong
        # to the earlier half; only strictly later observations are upper.
        lower = np.flatnonzero(times <= median)
        upper = np.flatnonzero(times > median)
        lower_span = median - float(np.min(times[lower])) if lower.size else 0.0
        upper_span = float(np.max(times[upper])) - median if upper.size else 0.0

        if upper.size and (not lower.size or upper_span >= lower_span):
            selected, side = upper, "late"
        else:
            selected, side = lower, "early"
        return selected, {
            "median": median,
            "side": side,
            "lower_span": lower_span,
            "upper_span": upper_span,
            "source_count": int(selected.size),
        }

    def _aug_client(
        self, train_X: np.ndarray, train_y: np.ndarray, k: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the data of a single client.

        Args:
            train_X: Feature data.
            train_y: Label data.
            k: Augmentation ratio.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented feature data and label data.
        """

        # Check that k is within (0, 1]
        if k <= 0 or k > 1:
            raise ValueError("k must be in the range (0, 1]")

        uncensor_num = int(np.sum(train_y[:, 1] == 1))
        sample_num = int(uncensor_num * k)
        if sample_num == 0:
            self.last_sampling_region = None
            return (
                np.empty((0, train_X.shape[1]), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        # Only uncensored data is used for training
        mask = train_y[:, 1] == 1
        vae = vae_train(
            train_X[mask],
            train_y[mask, 0],
            latent_num=self.latent_num,
            hidden_num=self.hidden_num,
            alpha=self.alpha,
            beta=self.beta,
        )

        if self.sampling == "unconditional":
            self.last_sampling_region = None
            return vae.sample(sample_num)

        sparse_index, region = self._sparse_region(train_y[mask, 0])
        self.last_sampling_region = region
        return vae.condition_sample(sparse_index, sample_num, gamma=self.sparse_gamma)

    def _check_clients_set(self, clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Validate the client datasets."""
        # Check that clients_set is not empty
        if not clients_set:
            raise ValueError("clients_set cannot be empty")

        # Each client must have at least 10 samples
        for i in clients_set.keys():
            if clients_set[i][0].shape[0] < 10:
                raise ValueError("each client must have at least 10 samples")

        # Each client must have at least one uncensored sample
        for i in clients_set.keys():
            if np.sum(clients_set[i][1][:, 1] == 1) == 0:
                raise ValueError("each client must have at least one uncensored sample")

    def mvaes(
        self, clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]], k: float = 1.0
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        MVAES (Multi-task Variational Autoencoder at the Server) method.
        Collect the augmented data from all clients at the server, then
        redistribute it to each client.

        Args:
            clients_set: Client datasets.
            k: Augmentation ratio.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Mapping of client ID to
                (augmented dataset, original + augmented dataset).
        """

        self._check_clients_set(clients_set)

        # Collect the augmented data from all clients at the server
        first_features = next(iter(clients_set.values()))[0]
        all_aug_X, all_aug_y = np.empty(
            shape=(0, first_features.shape[1]), dtype=np.float32
        ), np.empty(shape=(0, 2), dtype=np.float32)
        self.aug_clients_set = {}
        self.raw_aug_clients_set = {}
        self.sampling_regions = {}
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]
            pre = self._aug_client(train_X, train_y, k)
            X_pre, y_pre = pre[0], pre[1]
            self.aug_clients_set[i] = (X_pre, y_pre)
            self.sampling_regions[i] = self.last_sampling_region
            all_aug_X = np.vstack((all_aug_X, X_pre))
            all_aug_y = np.vstack((all_aug_y, y_pre))

        # Redistribute the augmented data to each client
        # (proportional to each client's local number of uncensored samples)
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]

            target = int(np.sum(train_y[:, 1] == 1) * k)
            if target == 0:
                self.raw_aug_clients_set[i] = (train_X.copy(), train_y.copy())
                continue
            sample_index = random.sample(range(all_aug_X.shape[0]), target)

            X_pre, y_pre = all_aug_X[sample_index,], all_aug_y[sample_index,]
            raw_aug_X = np.vstack((train_X, X_pre))
            raw_aug_y = np.vstack((train_y, y_pre))
            self.raw_aug_clients_set[i] = (raw_aug_X, raw_aug_y)

        return self.raw_aug_clients_set

    def mvaec(
        self, clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]], k: float = 1.0
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        MVAEC (Multi-task Variational Autoencoder at the Client) method.
        Each client is augmented with the synthetic samples it generates itself.

        Args:
            clients_set: Client datasets.
            k: Augmentation ratio.

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Mapping of client ID to
                (augmented dataset, original + augmented dataset).
        """

        self._check_clients_set(clients_set)

        self.raw_aug_clients_set = {}
        self.sampling_regions = {}
        for i in clients_set.keys():
            train_X, train_y = clients_set[i][0], clients_set[i][1]

            pre = self._aug_client(train_X, train_y, k)
            X_pre, y_pre = pre[0], pre[1]
            self.sampling_regions[i] = self.last_sampling_region

            target = int(np.sum(train_y[:, 1] == 1) * k)
            if target == 0:
                self.raw_aug_clients_set[i] = (train_X.copy(), train_y.copy())
                continue
            sample_index = random.sample(range(X_pre.shape[0]), target)

            raw_aug_X = np.vstack((train_X, X_pre[sample_index,]))
            raw_aug_y = np.vstack((train_y, y_pre[sample_index,]))
            self.raw_aug_clients_set[i] = (raw_aug_X, raw_aug_y)

        return self.raw_aug_clients_set
