from pycox.models import PCHazard, LogisticHazard, DeepHitSingle, CoxPH, CoxTime, CoxCC
from pycox.evaluation import EvalSurv
from copy import deepcopy
import torch.optim as optim
from Models import model
import numpy as np
import torch
import argparse
import json


def center(conf, datasets):
    """
    Centralized PC-Hazard
    :param conf:
    :param datasets:
    :return:
    """
    x_train = datasets.train_data.astype('float32')
    x_test = datasets.test_data.astype('float32')

    conf = deepcopy(conf)
    # print(conf)
    if conf.model_type == 'PC-Hazard':
        # labtrans = PCHazard.label_transform(conf['num_durations'])
        labtrans = PCHazard.label_transform(conf['num_durations'], scheme='quantiles')
    elif conf.model_type == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(conf['num_durations'], scheme='quantiles')
    elif conf.model_type == 'DeepHit':
        labtrans = DeepHitSingle.label_transform(conf['num_durations'], scheme='quantiles')
    elif conf.model_type == 'CoxTime':
        labtrans = CoxTime.label_transform()
    elif conf.model_type in ['DeepSurv', 'CoxPH', 'CoxCC']:
        labtrans = None
    else:
        raise ValueError('model error')
    get_target = lambda df: (df[:, 0], df[:, 1])

    y_train = get_target(datasets.train_label)

    if conf.model_type in ['PC-Hazard', 'LogisticHazard', 'DeepHit', 'CoxTime']:
        y_train = labtrans.fit_transform(*y_train)
        conf['labtrans'] = labtrans
        conf['out_features'] = labtrans.out_features
    elif conf.model_type in ['DeepSurv', 'CoxPH', 'CoxCC']:
        conf.out_features = 1

    durations_test, events_test = get_target(datasets.test_label)
    net = model(conf)
    net = net.model_initial()
    optimizer = optim.Adam(net.parameters(), lr=conf.learning_rate, weight_decay=0.0)
    if conf.model_type == 'PC-Hazard':
        center_model = PCHazard(net, optimizer, duration_index=labtrans.cuts)
    elif conf.model_type == 'LogisticHazard':
        center_model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
    elif conf.model_type == 'DeepHit':
        center_model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
    elif conf.model_type in ['DeepSurv', 'CoxPH']:
        center_model = CoxPH(net, optimizer)
    elif conf.model_type == 'CoxTime':
        center_model = CoxTime(net, optimizer, labtrans=labtrans)
    elif conf.model_type == 'CoxCC':
        center_model = CoxCC(net, optimizer)
    else:
        raise ValueError('Model not supported.')

    if conf.model_type == 'PC-Hazard':
        center_model.fit(x_train, y_train, epochs=conf.global_epochs, verbose=False,
                         batch_size=datasets.train_data.shape[0],
                         check_out_features=False)
    else:
        center_model.fit(x_train, y_train, epochs=conf.global_epochs, verbose=False,
                         batch_size=datasets.train_data.shape[0])

    if conf.model_type in ['DeepSurv', 'CoxPH', 'CoxTime', 'CoxCC']:
        _ = center_model.compute_baseline_hazards()  # Compute baseline hazards

    surv = center_model.predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    return [ev.concordance_td(), ev.integrated_brier_score(time_grid)]


if __name__ == '__main__':
    # np.random.seed(1234)
    # _ = torch.manual_seed(123)
    from federated_survival.data.generator import DataGenerator, SimulationConfig
    from federated_survival.data.splitter import DataSplitter
    from federated_survival.core.runner import FSARunner
    from federated_survival.core.config import FSAConfig

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
    DataSet = splitter.split(data_sdgm1)

    # Show the time range of each client
    for client_id, client_data in DataSet.clients_set.items():
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

    center_results = center(config, DataSet)
