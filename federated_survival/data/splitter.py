import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, NamedTuple
from sklearn.model_selection import train_test_split


class DataSet(NamedTuple):
    """Dataset containing clients_set, train_data, train_label, test_data, test_label, and raw_aug_clients_set"""
    clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]
    train_data: np.ndarray
    train_label: np.ndarray
    test_data: np.ndarray
    test_label: np.ndarray
    raw_aug_clients_set: Dict[str, Tuple[np.ndarray, np.ndarray]]


class DataSplitter:
    """Data splitter supporting IID, random, non-IID (censoring-based), Dirichlet, and Time-Non-IID partitioning"""
    
    def __init__(self, 
                 n_clients: int,
                 split_type: str = 'iid',
                 alpha: float = 0.5,
                 test_size: float = 0.2,
                 random_state: Optional[int] = None):
        """
        Initialize the data splitter

        Args:
            n_clients: Number of clients
            split_type: Partitioning type, one of 'iid', 'random', 'non-iid',
                'censoring-non-iid', 'time-non-iid', 'dirichlet'
            alpha: Dirichlet distribution parameter controlling the degree of non-IID heterogeneity
            test_size: Proportion of the test set
            random_state: Random seed
        """
        self.n_clients = n_clients
        self.split_type = split_type.lower()
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state

        if self.n_clients <= 0:
            raise ValueError("n_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.split_type not in [
            'iid', 'random', 'non-iid', 'censoring-non-iid',
            'time-non-iid', 'dirichlet'
        ]:
            raise ValueError(
                "split_type must be one of 'iid', 'random', 'non-iid', "
                "'censoring-non-iid', 'time-non-iid', 'dirichlet'"
            )
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def split(self, data: pd.DataFrame) -> DataSet:
        """
        Partition the data

        Args:
            data: Input data in the same format as produced by DataGenerator

        Returns:
            DataSet: Dataset containing clients_set, test_data, test_label, and raw_aug_clients_set
        """
        # First split into train/test sets, stratified by censoring status
        train_data, test_data = train_test_split(
            data, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=data['status']  # stratify by censoring status
        )

        # Convert to float32
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)
        
        # Assign data according to the partitioning scheme
        if self.split_type == 'iid':
            client_data = self._split_iid(train_data)
        elif self.split_type == 'random':
            client_data = self._split_random(train_data)
        elif self.split_type in ['non-iid', 'censoring-non-iid']:
            client_data = self._split_censoring_non_iid(train_data)
        elif self.split_type == 'dirichlet':
            # Use class attributes as default parameter values
            client_data = self._split_Dirichlet(
                train_data, 
                num_of_clients=self.n_clients, 
                beta=self.alpha, 
                n_time_bins=5  # default to 5 time bins
            )
        else:  # time-non-iid
            client_data = self._split_time_non_iid(train_data)

        # The unified seven-model workflow contains Cox-type objectives whose
        # event sampler is undefined on a client with no observed event.  Keep
        # the requested split as intact as possible by moving one event from a
        # donor only when a generated partition is degenerate.
        client_data = self._ensure_minimum_one_event(client_data)
        
        # Assign data to each client
        clients_set = {}
        for client_id, client_train_data in client_data.items():
            # Separate features and labels
            feature_cols = [col for col in client_train_data.columns if col not in ['time', 'status']]
            X = client_train_data[feature_cols].values
            y = client_train_data[['time', 'status']].values
            clients_set[f'client{client_id}'] = (X, y)
        
        # Prepare training data
        train_X = train_data[feature_cols].values
        train_y = train_data[['time', 'status']].values
        
        # Prepare test data
        test_X = test_data[feature_cols].values
        test_y = test_data[['time', 'status']].values
        
        # Initialize raw_aug_clients_set as an empty dict
        raw_aug_clients_set = {}
        
        return DataSet(
            clients_set=clients_set,
            train_data=train_X,
            train_label=train_y,
            test_data=test_X,
            test_label=test_y,
            raw_aug_clients_set=raw_aug_clients_set
        )

    def _ensure_minimum_one_event(
        self, client_data: Dict[int, pd.DataFrame]
    ) -> Dict[int, pd.DataFrame]:
        """Repair a generated partition so every client has one event.

        This is a conditioning constraint for a benchmark that must run Cox
        objectives on every client.  It is not an IID-preserving operation and
        should be reported as such for Dirichlet experiments.
        """
        total_events = int(sum(frame['status'].sum() for frame in client_data.values()))
        if total_events < self.n_clients:
            raise ValueError(
                "Cannot create non-degenerate client partitions: observed "
                f"events ({total_events}) are fewer than clients ({self.n_clients})"
            )

        repaired = {key: value.copy() for key, value in client_data.items()}
        missing = [key for key, frame in repaired.items() if int(frame['status'].sum()) == 0]
        for recipient in missing:
            event_counts = {
                key: int(frame['status'].sum()) for key, frame in repaired.items()
            }
            donor = max(event_counts, key=event_counts.get)
            if event_counts[donor] <= 1:
                raise ValueError("Unable to repair a zero-event client without creating another")
            donor_events = repaired[donor][repaired[donor]['status'] == 1]
            moved = donor_events.iloc[[0]].copy()
            repaired[donor] = repaired[donor].drop(index=moved.index).reset_index(drop=True)
            repaired[recipient] = pd.concat(
                [repaired[recipient], moved], ignore_index=True
            )

        return {key: frame.reset_index(drop=True) for key, frame in repaired.items()}
    
    def _split_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """IID partitioning that keeps the censoring rate identical across clients"""
        # Get feature columns
        feature_cols = [col for col in data.columns if col not in ['time', 'status']]
        # Get the censoring column
        status_col = [col for col in data.columns if col.startswith('status')][0]
        # Stratify by the censoring column
        data_0 = data[data[status_col] == 0]
        data_1 = data[data[status_col] == 1]
        n_samples_0 = len(data_0)
        n_samples_1 = len(data_1)
        samples_per_client_0 = n_samples_0 // self.n_clients
        samples_per_client_1 = n_samples_1 // self.n_clients
        
        client_data = {}
        for i in range(self.n_clients):
            start_idx_0 = i * samples_per_client_0
            end_idx_0 = (i + 1) * samples_per_client_0 if i < self.n_clients - 1 else n_samples_0
            start_idx_1 = i * samples_per_client_1
            end_idx_1 = (i + 1) * samples_per_client_1 if i < self.n_clients - 1 else n_samples_1
            client_data[i] = pd.concat([data_0.iloc[start_idx_0:end_idx_0].copy(), data_1.iloc[start_idx_1:end_idx_1].copy()])
        
        return client_data
    
    def _split_random(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Unstratified random split (still IID in expectation)."""
        # Shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)
        
        n_samples = len(data)
        samples_per_client = n_samples // self.n_clients
        client_data = {}
        for i in range(self.n_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.n_clients - 1 else n_samples
            client_data[i] = data.iloc[start_idx:end_idx].copy()
        return client_data

    def _split_censoring_non_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Create a reproducible censoring-rate shift across clients."""
        rng = np.random.RandomState(self.random_state)
        event = data[data['status'] == 1].sample(
            frac=1, random_state=self.random_state
        )
        censored = data[data['status'] == 0].sample(
            frac=1,
            random_state=None if self.random_state is None else self.random_state + 1,
        )
        increasing = np.linspace(1.0, 3.0, self.n_clients)
        event_counts = rng.multinomial(len(event), increasing / increasing.sum())
        censor_counts = rng.multinomial(
            len(censored), increasing[::-1] / increasing.sum()
        )

        def partition(frame, counts):
            pieces, start = [], 0
            for count in counts:
                pieces.append(frame.iloc[start:start + count])
                start += count
            return pieces

        event_parts = partition(event, event_counts)
        censor_parts = partition(censored, censor_counts)
        result = {}
        for client_id in range(self.n_clients):
            client = pd.concat(
                [event_parts[client_id], censor_parts[client_id]], ignore_index=True
            )
            result[client_id] = client.sample(
                frac=1,
                random_state=None if self.random_state is None else self.random_state + client_id,
            )
        return result


    def _split_Dirichlet(self, data: pd.DataFrame, num_of_clients: int, beta: float, n_time_bins: int) -> Dict[int, pd.DataFrame]:
        """
        Dirichlet-based non-IID partitioning over composite classes formed by
        time binning plus event status. The `time` column is divided into
        n_time_bins intervals and combined with status (0/1) into
        n_time_bins * 2 pseudo-classes. Dirichlet(beta) allocation is applied
        independently per pseudo-class, yielding joint time + event heterogeneity.

        Args:
            data (pd.DataFrame): DataFrame containing 'time' and 'status' columns.
            num_of_clients (int): Number of clients
            beta (float): Concentration parameter of the Dirichlet distribution (smaller means more non-IID)
            n_time_bins (int): Number of time bins (3-5 recommended)

        Returns:
            Dict[int, pd.DataFrame]: Mapping from client ID to the sub-DataFrame assigned to that client.
        """
        # 1. Bin the time column
        data = data.copy()
        data['time_bin'] = pd.cut(data['time'], bins=n_time_bins, labels=False)
        
        # 2. Build composite pseudo-classes: time_bin * 2 + status
        data['pseudo_label'] = data['time_bin'] * 2 + data['status'].astype(int)
        
        # 3. Collect all unique pseudo-classes
        pseudo_labels = data['pseudo_label'].values
        unique_pseudo_labels = np.sort(data['pseudo_label'].unique())
        n_pseudo_labels = len(unique_pseudo_labels)
        
        # 4. Sample allocation proportions per pseudo-class from the Dirichlet distribution
        #    shape: (n_pseudo_labels, num_of_clients)
        pseudo_label_proportions = np.random.dirichlet(
            alpha=[beta] * num_of_clients,
            size=n_pseudo_labels
        )

        # 5. Initialize the index list of each client
        client_indices = [[] for _ in range(num_of_clients)]

        # 6. Iterate over each pseudo-class and allocate its samples
        for pl_idx, pl_value in enumerate(unique_pseudo_labels):
            # Find all sample indices belonging to the current pseudo-class
            pl_mask = (pseudo_labels == pl_value)
            pl_sample_indices = np.where(pl_mask)[0]
            n_pl_samples = len(pl_sample_indices)

            # Skip this pseudo-class if it has no samples
            if n_pl_samples == 0:
                continue

            # Get the proportions allocated to each client for this pseudo-class
            proportions = pseudo_label_proportions[pl_idx]

            # Use a multinomial draw to assign sample counts to clients based on
            # the proportions; this keeps the allocation discrete and the total
            # equal to the number of samples
            assigned_counts = np.random.multinomial(n_pl_samples, proportions)

            # Randomly shuffle the sample indices of the current pseudo-class for randomness
            np.random.shuffle(pl_sample_indices)

            # Split and assign the sample indices to clients by allocated count
            start = 0
            for client_id, count in enumerate(assigned_counts):
                if count > 0:
                    end = start + count
                    client_indices[client_id].extend(pl_sample_indices[start:end])
                    start = end

        # 7. Build the data dict of each client from the collected indices
        client_data = {}
        for client_id in range(num_of_clients):
            # Fetch the samples assigned to this client and drop temporary columns
            client_samples = data.iloc[client_indices[client_id]].copy()
            client_samples = client_samples.drop(columns=['time_bin', 'pseudo_label'])
            client_data[client_id] = client_samples.reset_index(drop=True)

        return client_data
    
    def _split_time_non_iid(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Time-Non-IID partitioning based on survival time, distinguishing censoring status"""
        # Get feature columns
        feature_cols = [col for col in data.columns if col not in ['time', 'status']]
        # Get the censoring column
        status_col = [col for col in data.columns if col.startswith('status')][0]
        # Stratify by the censoring column
        data_0 = data[data[status_col] == 0]
        data_1 = data[data[status_col] == 1]
        n_samples_0 = len(data_0)
        n_samples_1 = len(data_1)
        
        # Sort by survival time
        sorted_indices_0 = data_0['time'].sort_values().index
        sorted_indices_1 = data_1['time'].sort_values().index
        
        # Divide the time range into n_clients intervals
        time_ranges_0 = np.array_split(sorted_indices_0, self.n_clients)
        time_ranges_1 = np.array_split(sorted_indices_1, self.n_clients)
        
        # Sample each time interval using the Dirichlet distribution
        client_data = {}
        for i in range(self.n_clients):
            # Get the samples in the current time interval
            time_range_indices_0 = time_ranges_0[i]
            time_range_indices_1 = time_ranges_1[i]
            time_range_data_0 = data_0.loc[time_range_indices_0]
            time_range_data_1 = data_1.loc[time_range_indices_1]

            client_data[i] = pd.concat([time_range_data_0, time_range_data_1])
        
        return client_data 
