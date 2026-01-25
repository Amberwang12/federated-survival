# -*- coding: UTF-8 -*-
import time
# import torch
# import numpy as np
import pandas as pd
from central import center
from client_local import clients_local
from copy import deepcopy
from federated_survival.data.generator import DataGenerator, SimulationConfig
from federated_survival.data.splitter import DataSplitter
from federated_survival.core.runner import FSARunner
from federated_survival.core.config import FSAConfig
import torch.nn as nn

# np.random.seed(1110)
# _ = torch.manual_seed(1110)


def compare_fed_center(conf, datasets):
    conf = deepcopy(conf)

    raw_result = []
    # print('#### train raw data federated ####')
    model_name = conf.model_type

    # Initialize and run the federated learning process
    runner = FSARunner(conf)
    results = runner.run(datasets)

    # Access evaluation metrics
    test_cindex = results['test_Cindex'][-1]
    test_ibs = results['test_IBS'][-1]
    fed_result = [test_cindex, test_ibs]  

    # federated learning with augmentation MVAEC
    runner = FSARunner(conf)
    results = runner.run(
        datasets,
        type='raw_aug',
        aug_method='MVAEC'  # or 'MVAES'
    )

    # Access evaluation metrics
    test_cindex = results['test_Cindex'][-1]
    test_ibs = results['test_IBS'][-1]
    mvaec_result = [test_cindex, test_ibs]  


    # federated learning with augmentation MVAES
    runner = FSARunner(conf)
    results = runner.run(
        datasets,
        type='raw_aug',
        aug_method='MVAES'  # or 'MVAES'
    )

    # Access evaluation metrics
    test_cindex = results['test_Cindex'][-1]
    test_ibs = results['test_IBS'][-1]
    mvaes_result = [test_cindex, test_ibs]  

    # federated learning with differential privacy
    # Gaussian mechanism parameters
    conf.use_differential_privacy = True
    conf.dp_mechanism = 'gaussian'
    conf.dp_epsilon = 10
    conf.dp_delta = 1e-4
    conf.dp_sensitivity = 0.5
    conf.dp_noise_multiplier = 1.0
    conf.dp_clip_norm = 2.0
    conf.verbose = True

    runner = FSARunner(conf)
    results = runner.run(
        datasets,
    )

    # Access evaluation metrics
    test_cindex = results['test_Cindex'][-1]
    test_ibs = results['test_IBS'][-1]
    fed_dp_result = [test_cindex, test_ibs]  

    print('#### train raw data center for {} ####'.format(model_name))
    center_result = center(conf, datasets)

    print('#### train raw data local for {} ####'.format(model_name))
    local_result = clients_local(conf, datasets)

    metrics = []
    for j in [fed_result,
              mvaec_result, mvaes_result,
              fed_dp_result,
              center_result, local_result]:
        metrics = metrics + j
    return metrics


def simulate_data(conf):
    for _ in range(3):
        # 重试机制
        try:
            result = []
            i = 1
            for j in range(100):
                # Configure data generation
                sim_config = SimulationConfig(
                    n_samples=100,  # Number of samples
                    n_features=10,  # Number of features
                    random_state=j  # Random seed for reproducibility
                )

                # Generate simulated data
                generator = DataGenerator(config=sim_config)
                datasets = generator.generate('SDGM1', c_mean=0.4)  # Standard proportional hazards

                sum(datasets['status'])

                # Initialize splitter with specific configuration
                splitter = DataSplitter(
                    n_clients=3,  # Number of federated learning clients
                    split_type='Dirichlet',  # Partition type: 'iid', 'non-iid', 'time-non-iid', 'Dirichlet'
                    alpha=0.8,  # Dirichlet distribution parameter for non-IID splitting
                    test_size=0.2,  # Proportion of test set
                    random_state=j  # Random seed for reproducibility
                )

                # Split and distribute data to clients
                dataSet = splitter.split(datasets)

                print(f'------the {j+1}-th repeat------')

                result.append(compare_fed_center(conf, dataSet))
                i += 1
                # break
                # break
        except Exception as e:
            print(e)
        else:
            return result



def experiment1(conf):
    """
    第一个实验：对比四种模拟数据,三个删失率下标准联邦、VAE增强联邦、VAE+BJ增强联邦、集中式、local
    :param conf:
    :return:
    """
    # print(conf)  # 打印配置
    # print(id(conf))
    conf = deepcopy(conf)

    result = simulate_data(conf)
    conf.dataset_name = 'simulate'

    # if conf.mode == 'simulate':
    #     conf.target_censor_rate = censor_rate
    #     conf.c_mean = conf.c_mean_dict[conf.sim_type][conf.target_censor_rate]
    result = simulate_data(conf)
    conf.dataset_name = 'simulate'
    conf.split_methods = 'Dirichlet'
    # elif conf.mode == 'real':
    #     pass
        # result = real_data(conf)

    # print(result)
    result_df = pd.DataFrame(result, columns=[
                                              'fed-Cindex', 'fed-IBS',
                                              'mvaec-Cindex', 'mvaec-IBS',
                                              'mvaes-Cindex', 'mvaes-IBS',
                                              'fed-dp-Cindex', 'fed-dp-IBS',
                                              'center-Cindex', 'center-IBS'] +
                                             ['client' + str(i + 1) + '-' + j for i in
                                              range(3) for j in
                                              ['Cindex', 'IBS']] +
                                             ['clients-mean-Cindex', 'clients-mean-IBS'])
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(result_df.mean())
    print(conf)
    print(cur_time)
    with open('result/' + conf.model_type + '_' + conf.dataset_name + '_' + conf.split_methods + '_' +
              cur_time + '_federal_and_central-result.txt', 'w') as f:  # 设置文件对象
        print(conf, file=f)
    result_df.to_csv('result/' + conf.model_type + '_' + conf.dataset_name + '_' + conf.split_methods + '_' +
                     cur_time + '_federal_and_central-result.csv',
                     encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    # Configure the federated learning process
    config = FSAConfig(
        num_clients=3,  # Number of federated learning clients
        n_features=10,  # Number of features
        n_samples=100,  # Number of samples
        model_type='DeepSurv',  # Survival model type
        num_nodes=(32, 32),
        dropout=True,
        local_epochs=2,  # Number of local training epochs
        global_epochs=10,  # Number of global communication rounds
        learning_rate=0.01,  # Learning rate
        batch_size=16,  # Batch size
        random_seed=42,  # Random seed
        client_sample_ratio=1,  # Ratio of clients selected in each round
        early_stopping=True,  # Enable early stopping
        early_stopping_patience=5,  # Number of epochs to wait before early stopping
        # Augmentation parameters
        latent_num=5,  # Dimension of latent space
        hidden_num=32,  # Dimension of hidden layer
        alpha=1.0,  # Weight for KL divergence
        beta=1.0,  # Weight for conditional loss
        k=0.5  # Augmentation ratio (0 < k <= 1)
    )

    experiment1(config)
