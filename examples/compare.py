# -*- coding: UTF-8 -*-
import time
from clients import *
from getData import GetDataSet
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from FedAvg import Fedmodel
from central import center
from client_local import clients_local
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid
from copy import deepcopy

np.random.seed(1110)
_ = torch.manual_seed(1110)

def compare_fed_center(conf, datasets):
    conf = deepcopy(conf)
    datasets.preprocess()  # 标准化
    # conf['aug_method'] = 'VAE'
    if conf['split_methods'] == 'iid':
        datasets.dataSetBalanceAllocation(num_of_clients=conf['num_of_clients'])  # 分客户端
        # datasets.aug_data_all(conf['aug_method'])  # 数据增强
    elif conf['split_methods'] == 'dirichlet':
        datasets.dataSetAllocation_noniid_dirichlet(num_of_clients=conf['num_of_clients'],
                                                    beta=conf['dirichlet_beta'])
    elif conf['split_methods'] == 'time-noniid':
        datasets.dataSetBalanceAllocation_time_noniid(num_of_clients=conf['num_of_clients'])
        # datasets.aug_data_patial(conf['censor_bound'])
    # censor_rate[i] = {'raw': datasets.check_censoring_rate(type='raw'),
    #                   'raw_aug': datasets.check_censoring_rate(type='raw_aug')}
    conf['input_shape'] = datasets.feature_num

    raw_result = []
    # print('#### train raw data federated ####')
    # for model_name in ['CoxPH-linear', 'CoxPH',
    #                    'DeepHit', 'CoxTime', 'CoxCC',
    #                    'LogisticHazard',
    #                    'PC-Hazard']:
    model_name = conf['model']
    print('#### train raw data federated for {} ####'.format(model_name))
    # conf['model'] = model_name
    if conf['model'] == 'CoxPH-linear':
        # cox_conf = deepcopy(conf)
        conf['model'] = 'CoxPH'
        conf['num_nodes'] = []
        conf['batch_norm'] = False

    sub_raw_result = Fedmodel(conf, datasets, type='raw')

    raw_result = raw_result + sub_raw_result

    print('#### train raw and aug global data federated for VAE ####')
    conf['aug_method'] = 'VAE'
    datasets.aug_data_global(conf['aug_method'], k=conf['k'], alloc_method=conf['alloc_method'])
    raw_vae_global_aug_result = Fedmodel(conf, datasets, type='raw_aug')
    print('#### train raw and aug local data federated for VAE ####')
    datasets.aug_data_local(conf['aug_method'], k=conf['k'])
    raw_vae_local_aug_result = Fedmodel(conf, datasets, type='raw_aug')

    # print('#### train raw and aug global data federated for VAE+BJ ####')
    # conf['aug_method'] = 'VAE+BJ'
    # datasets.aug_data_global(conf['aug_method'], k=conf['k'], alloc_method=conf['alloc_method'])
    # raw_vaebj_global_aug_result = Fedmodel(conf, datasets, type='raw_aug')
    # print('#### train raw and aug local data federated for VAE+BJ ####')
    # datasets.aug_data_local(conf['aug_method'], k=conf['k'])
    # raw_vaebj_local_aug_result = Fedmodel(conf, datasets, type='raw_aug')

    print('#### train raw data center for {} ####'.format(model_name))
    center_result = center(conf, datasets)

    print('#### train raw data local for {} ####'.format(model_name))
    local_result = clients_local(conf, datasets)

    metrics = []
    for j in [raw_result,
              raw_vae_global_aug_result, raw_vae_local_aug_result,
              # raw_vaebj_global_aug_result, raw_vaebj_local_aug_result,
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
                datasets = GetDataSet(conf)
                skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=j)
                X, y = datasets.X, datasets.y
                for train_index, test_index in skf.split(X, y['status']):
                    print(f'------the {i}-th cross validation------')
                    datasets.train_data, datasets.train_label = X.iloc[train_index, :], y.iloc[train_index, :].values
                    datasets.test_data, datasets.test_label = X.iloc[test_index, :], y.iloc[test_index, :].values
                    result.append(compare_fed_center(conf, datasets))
                    i += 1
                    break
                # break
        except Exception as e:
            print(e)
        else:
            return result


def real_data(conf):
    for _ in range(3):
        # 重试机制
        try:
            result = []
            datasets = GetDataSet(conf)
            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=12345)
            i = 1
            X, y = datasets.X, datasets.y
            # censor_rate = {}
            for train_index, test_index in skf.split(X, y['status']):
                print(f'------the {i}-th cross validation------')
                datasets.train_data, datasets.train_label = X.iloc[train_index, :], y.iloc[train_index, :].values
                datasets.test_data, datasets.test_label = X.iloc[test_index, :], y.iloc[test_index, :].values
                result.append(compare_fed_center(conf, datasets))
                i += 1
                # break
        except Exception as e:
            print(e)
        else:
            return result


def experiment1(conf, censor_rate='40%'):
    """
    第一个实验：对比四种模拟数据,三个删失率下标准联邦、VAE增强联邦、VAE+BJ增强联邦、集中式、local
    :param conf:
    :return:
    """
    # print(conf)  # 打印配置
    # print(id(conf))
    conf = deepcopy(conf)

    if conf['mode'] == 'simulate':
        conf['target_censor_rate'] = censor_rate
        conf['c_mean'] = conf['c_mean_dict'][conf['sim_type']][conf['target_censor_rate']]
        result = simulate_data(conf)
        conf['dataset_name'] = 'simulate'
    elif conf['mode'] == 'real':
        result = real_data(conf)

    # print(result)
    result_df = pd.DataFrame(result, columns=[
                                              #    'FL-CoxPH-Cindex', 'FL-CoxPH-IBS',
                                              # 'FL-DeepSurv-Cindex', 'FL-DeepSurv-IBS',
                                              # 'FL-DeepHit-Cindex', 'FL-DeepHit-IBS',
                                              # 'FL-CoxTime-Cindex', 'FL-CoxTime-IBS',
                                              # 'FL-CoxCC-Cindex', 'FL-CoxCC-IBS',
                                              # 'FL-LogisticHazard-Cindex', 'FL-LogisticHazard-IBS',
                                              'FL-Cindex', 'FL-IBS',
                                              # 'DeepHit', 'CoxTime', 'CoxCC',
                                              # 'LogisticHazard', 'PC-Hazard'
                                              'FL-vae-global-aug-Cindex', 'FL-vae-global-aug-IBS',
                                              'FL-vae-local-aug-Cindex', 'FL-vae-local-aug-IBS',
                                              # 'FL-vaebj-global-aug-Cindex', 'FL-vaebj-global-aug-IBS',
                                              # 'FL-vaebj-local-aug-Cindex', 'FL-vaebj-local-aug-IBS',
                                              'center-Cindex', 'center-IBS'] +
                                             ['client' + str(i + 1) + '-' + j for i in
                                              range(conf["num_of_clients"]) for j in
                                              ['Cindex', 'IBS']] +
                                             ['clients-mean-Cindex', 'clients-mean-IBS'])
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(result_df.mean())
    print(conf)
    print(cur_time)
    with open('result/' + conf['model'] + '_' + conf['dataset_name'] + '_' + conf['split_methods'] + '_' +
              conf['sim_type'] + '_' + conf['target_censor_rate'] + '_' +
              # str(conf['dirichlet_beta']) + '_' + str(conf['censor_bound']) +
              cur_time + '_federal_and_central-result.txt', 'w') as f:  # 设置文件对象
        print(conf, file=f)
    result_df.to_csv('result/' + conf['model'] + '_' + conf['dataset_name'] + '_' + conf['split_methods'] + '_' +
                     conf['sim_type'] + '_' + conf['target_censor_rate'] + '_' +
                     # str(conf['dirichlet_beta']) + '_' + str(conf['censor_bound']) +
                     cur_time + '_federal_and_central-result.csv',
                     encoding='utf_8_sig', index=False)


def pool_experiment1(conf):
    """
    多进程运行实验1,不需要指定删失率，只需要指定数据集
    :param conf:
    :return:
    """
    pool = Pool(4)
    if conf['mode'] == 'simulate':
        for i in conf['c_mean_dict'][conf['sim_type']].keys():
            print(i)
            pool.apply_async(experiment1, (conf, i))
    elif conf['mode'] == 'real':
        pool.apply_async(experiment1, (conf, ))
    pool.close()
    pool.join()