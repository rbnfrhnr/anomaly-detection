import math
from glob import glob

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import utils


def load(cfg):
    logger = utils.get_logger(**cfg)
    data_location = cfg['data']['location']
    train_sets = cfg['data']['train-sets']
    dataset_nr = train_sets[0]
    t_steps = cfg['preprocessing']['time-steps']

    augmentations = []
    if 'preprocessing' in cfg:
        augmentations = cfg['preprocessing']['augmentation'] if 'augmentation' in cfg['preprocessing'] else []

    file = glob(data_location + '/' + str(dataset_nr) + '*')[0]
    file_name = file.split('/')[-1]
    data = np.fromfile(file, sep='\n').reshape(-1, 1)
    dataset_name = file_name.split('.')[0]
    dataset_info = dataset_name.split('_')
    train_from = 0
    train_to = int(dataset_info[4])
    test_from = train_to
    anomaly_from = int(dataset_info[5])
    anomaly_to = int(dataset_info[6])
    anomaly_range = np.arange(anomaly_from, anomaly_to)

    if t_steps == 'dynamic':
        window_size = math.ceil((anomaly_to - anomaly_from) * 0.75)
    else:
        window_size = t_steps
    logger.info('window size: ' + str(window_size))
    cfg['preprocessing']['time-steps'] = window_size

    train_data = data[:train_to]
    train_data = train_data[:math.floor(train_data.shape[0] / window_size) * window_size]
    train_data_normalized = MinMaxScaler().fit_transform(train_data)
    train_data_normalized = train_data_normalized.reshape(-1, window_size, 1)

    test_data = data[train_to:]
    test_data = test_data[:math.floor(test_data.shape[0] / window_size) * window_size]
    test_data_normalized = MinMaxScaler().fit_transform(test_data)
    test_data_idx = np.arange(test_data.shape[0])
    new_anomaly_from = anomaly_from - train_to
    new_anomaly_to = anomaly_to - train_to
    new_anomaly_range = np.arange(new_anomaly_from, new_anomaly_to)
    test_y = np.isin(test_data_idx, new_anomaly_range).astype(int)

    # will not be relevant for ucr dataset. only here for the kde to fit an anomalicious pdf
    # which will not be used for the identificaiton of anomalies
    dummy_train_mal = np.ones((1, window_size, 1))
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(train_data.reshape(-1).shape[0] - window_size - 1)[:,
                                                    None]
    train_data_normalized = train_data_normalized.reshape(-1)[indexer]
    train_data_normalized = train_data_normalized.reshape(train_data_normalized.shape[0], window_size, 1)
    base = train_data_normalized.copy()
    for augmentation in augmentations:
        train_data_normalized = np.concatenate([base, utils.augment2(base, augmentation)])

    return train_data_normalized, dummy_train_mal, {'real-test': (test_data_normalized, test_y)}


def load2(cfg):
    logger = utils.get_logger(**cfg)
    data_location = cfg['data']['location']
    train_sets = cfg['data']['train-sets']
    dataset_nr = train_sets[0]
    window_size = cfg['preprocessing']['time-steps']

    augmentations = []
    if 'preprocessing' in cfg:
        augmentations = cfg['preprocessing']['augmentation'] if 'augmentation' in cfg['preprocessing'] else []

    file = glob(data_location + '/' + str(dataset_nr) + '*')[0]
    file_name = file.split('/')[-1]
    data = np.fromfile(file, sep='\n').reshape(-1, 1)
    dataset_name = file_name.split('.')[0]
    dataset_info = dataset_name.split('_')
    train_from = 0
    train_to = int(dataset_info[4])
    test_from = train_to
    anomaly_from = int(dataset_info[5])
    anomaly_to = int(dataset_info[6])
    anomaly_range = np.arange(anomaly_from, anomaly_to)

    if window_size == 'dynamic':
        periodicity = utils.get_periodicity(data[:train_to])
        window_size = np.round(window_size)
        cfg['preprocessing']['time-steps-old'] = 'dynamic'

    cfg['preprocessing']['time-steps'] = int(window_size)

    train_data = data[:train_to]
    train_data = train_data[:math.floor(train_data.shape[0] / window_size) * window_size]
    train_data_normalized = MinMaxScaler().fit_transform(train_data)
    train_data_normalized = train_data_normalized.reshape(-1, window_size)
    test_data = data[train_to:]
    test_data = test_data[:math.floor(test_data.shape[0] / window_size) * window_size]
    test_data_normalized = MinMaxScaler().fit_transform(test_data)
    test_data_idx = np.arange(test_data.shape[0])
    new_anomaly_from = anomaly_from - train_to
    new_anomaly_to = anomaly_to - train_to
    new_anomaly_range = np.arange(new_anomaly_from, new_anomaly_to)
    test_y = np.isin(test_data_idx, new_anomaly_range).astype(int)
    # will not be relevant for ucr dataset. only here for the kde to fit an anomalicious pdf
    # which will not be used for the identificaiton of anomalies
    dummy_train_mal = np.ones((1, window_size, 1))
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(train_data.reshape(-1).shape[0] - window_size - 1)[:,
                                                    None]
    #     indexer = np.arange(train_data.reshape(-1).shape[0]).reshape(-1, window_size)
    print(indexer)
    train_data_normalized = train_data_normalized.reshape(-1)[indexer]
    train_data_normalized = train_data_normalized.reshape(train_data_normalized.shape[0], window_size)

    test_y = test_y.reshape(-1, window_size)
    y_2 = test_y.sum(axis=1)
    y_2 = y_2 >= 1
    y_2 = y_2.reshape(-1)

    base = train_data_normalized.copy()
    for augmentation in augmentations:
        train_data_normalized = np.concatenate([base, utils.augment2(base, augmentation)])

    # return train_data_normalized, test_data_normalized.reshape(-1, window_size), y_2

    return train_data_normalized, dummy_train_mal, {'real-test': (test_data_normalized, test_y)}
