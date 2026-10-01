import argparse
import json
from central import center
import numpy as np
from copy import deepcopy
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter, DataSet
from federated_survival.core.runner import FSARunner
from federated_survival.core.config import FSAConfig


def clients_local(conf, datasets, type='raw'):
    """
    :param type:
    :param conf:
    :param datasets:
    :return: Cindex and IBS for each locally trained client, plus the Cindex and IBS averaged over all clients
    """
    conf = deepcopy(conf)
    if type == 'raw':
        clients_sets = datasets.clients_set
    elif type == 'raw_aug':
        clients_sets = datasets.raw_aug_clients_set
    else:
        exit()
    print(f'There are {len(clients_sets)} clients')

    metrics = []
    for i in clients_sets.keys():
        print(f'local train {i}')
        local_datasets = DataSet(clients_set={},
                                 train_data=clients_sets[i][0],
                                 train_label=clients_sets[i][1],
                                 test_data=datasets.test_data,
                                 test_label=datasets.test_label,
                                 raw_aug_clients_set={})
        client_metrics = center(conf, local_datasets)
        metrics.append(client_metrics)
    metrics.append(list(np.mean(metrics, axis=0)))  # Append the mean
    ret = []
    for j in metrics:
        ret = ret + j

    return ret


if __name__ == '__main__':

    # Configure data generation
    sim_config = SimulationConfig(
        n_samples=100,      # Number of samples
        n_features=10,      # Number of features
        random_state=42     # Random seed for reproducibility
    )

    # Generate simulated data
    generator = DataGenerator(config=sim_config)
    data_sdgm1 = generator.generate('SDGM1', c_mean=0.4)  # Standard proportional hazards

    # Initialize splitter with specific configuration
    splitter = DataSplitter(
        n_clients=3,           # Number of federated learning clients
        split_type='Dirichlet',      # Partition type: 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'
        alpha=0.8,             # Dirichlet distribution parameter for non-IID splitting
        test_size=0.2,         # Proportion of test set
        random_state=42        # Random seed for reproducibility
    )

    # Split and distribute data to clients
    dataSet = splitter.split(data_sdgm1)

    # Show the time range of each client
    for client_id, client_data in dataSet.clients_set.items():
        train_x, train_y = client_data
        test_durations = train_y[:, 0]
        print(f"Client {client_id}:")
        print(f"Time Range: [{test_durations.min():.2f}, {test_durations.max():.2f}]")
        print("\n")

    # Configure the federated learning process
    config = FSAConfig(
        num_clients=3,  # Number of federated learning clients
        n_features=10,  # Number of features
        n_samples=100,  # Number of samples
        model_type='DeepSurv',  # Survival model type
        local_epochs=1,  # Number of local training steps per round (tuned)
        global_epochs=30,  # Number of global communication rounds (tuned)
        learning_rate=0.003,  # Learning rate (tuned)
        batch_size=32,  # Batch size
        random_seed=42,  # Random seed
        client_sample_ratio=0.5,  # Ratio of clients selected in each round
        early_stopping=True,  # Enable early stopping
        early_stopping_patience=5  # Number of epochs to wait before early stopping
    )

    client_local_results = clients_local(config, dataSet)
