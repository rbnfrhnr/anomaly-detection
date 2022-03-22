import math

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import utils


def load_normal(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    return sigs


def load_windowed_no_overlap(cfg):
    data_location = cfg['data']['location']
    train_sets = cfg['data']['train-sets']
    # test_groups = cfg['data']['test-groups']
    t_steps = cfg['preprocessing']['time-steps']
    window_size = t_steps
    nr = train_sets[0]
    augmentations = []
    if 'preprocessing' in cfg:
        augmentations = cfg['preprocessing']['augmentation'] if 'augmentation' in cfg['preprocessing'] else []

    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])[0]
    sigs = sigs[:math.floor(sigs.shape[0] / window_size) * window_size]
    sigs = sigs.reshape(-1, window_size)
    mal_idxs = np.array(anot.sample)[(np.array(anot.symbol) != 'N').nonzero()[0]]
    indexer = np.arange(sigs.shape[0] * window_size).reshape(-1, window_size)
    mal = sigs[np.argwhere(np.isin(indexer, mal_idxs))[:, 0]]
    norm = sigs[np.delete(np.arange(sigs.shape[0]), np.argwhere(np.isin(indexer, mal_idxs))[:, 0])]

    norm_train, norm_test = train_test_split(norm, test_size=0.25)
    mal_train, mal_test = train_test_split(mal, test_size=0.25)

    normalized_train = MinMaxScaler().fit_transform(np.concatenate([norm_train, mal_train]))
    norm_train_normalized = normalized_train[0:norm_train.shape[0]]
    norm_train_normalized = norm_train_normalized.reshape(norm_train_normalized.shape[0],
                                                          norm_train_normalized.shape[1], 1)
    mal_train_normalized = normalized_train[norm_train.shape[0]:]
    mal_train_normalized = mal_train_normalized.reshape(mal_train_normalized.shape[0], mal_train_normalized.shape[1], 1)

    normalized_test = MinMaxScaler().fit_transform(np.concatenate([norm_test, mal_test]))
    norm_test_normalized = normalized_test[0:norm_test.shape[0]]
    norm_test_normalized = norm_test_normalized.reshape(norm_test_normalized.shape[0], norm_test_normalized.shape[1], 1)
    mal_test_normalized = normalized_test[norm_test.shape[0]:]
    mal_test_normalized = mal_test_normalized.reshape(mal_test_normalized.shape[0], mal_test_normalized.shape[1], 1)

    base = norm_train_normalized.copy()
    for augmentation in augmentations:
        norm_train_normalized = np.concatenate([base, utils.augment2(base, augmentation)])

    return norm_train_normalized, mal_train_normalized, {'real-test': (norm_test_normalized, mal_test_normalized)}


def load_sliding_window(cfg):
    data_location = cfg['data']['location']
    train_sets = cfg['data']['train-sets']
    # test_groups = cfg['data']['test-groups']
    t_steps = cfg['preprocessing']['time-steps']
    window_size = t_steps
    nr = train_sets[0]
    augmentations = []
    if 'preprocessing' in cfg:
        augmentations = cfg['preprocessing']['augmentation'] if 'augmentation' in cfg['preprocessing'] else []

    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])[0]

    norm_idxs = np.array(anot.sample)[(np.array(anot.symbol) == 'N').nonzero()[0]]
    norm_idxs_train, norm_idxs_test = train_test_split(norm_idxs, test_size=0.25)
    mal_idxs = np.array(anot.sample)[(np.array(anot.symbol) != 'N').nonzero()[0]]
    mal_idx_train, mal_idxs_test = train_test_split(mal_idxs, test_size=0.25)

    indexer = np.arange(window_size)[None, :] + 1 * np.arange(sigs.shape[0] - window_size - 1)[:, None]
    mal_train = sigs[indexer[np.isin(indexer, mal_idx_train).nonzero()[0]]]
    mal_test = sigs[indexer[np.isin(indexer, mal_idxs_test).nonzero()[0]]]
    norm_train = sigs[indexer[np.isin(indexer, norm_idxs_train).nonzero()[0]]]
    norm_test = sigs[indexer[np.isin(indexer, norm_idxs_test).nonzero()[0]]]

    normalized_train = MinMaxScaler().fit_transform(np.concatenate([norm_train, mal_train]))
    norm_train_normalized = normalized_train[0:norm_train.shape[0]]
    norm_train_normalized = norm_train_normalized.reshape(norm_train_normalized.shape[0],
                                                          norm_train_normalized.shape[1], 1)
    mal_train_normalized = normalized_train[norm_train.shape[0]:]
    mal_train_normalized = mal_train_normalized.reshape(mal_train_normalized.shape[0], mal_train_normalized.shape[1], 1)

    normalized_test = MinMaxScaler().fit_transform(np.concatenate([norm_test, mal_test]))
    norm_test_normalized = normalized_test[0:norm_test.shape[0]]
    norm_test_normalized = norm_test_normalized.reshape(norm_test_normalized.shape[0], norm_test_normalized.shape[1], 1)
    mal_test_normalized = normalized_test[norm_test.shape[0]:]
    mal_test_normalized = mal_test_normalized.reshape(mal_test_normalized.shape[0], mal_test_normalized.shape[1], 1)

    base = norm_train_normalized.copy()
    for augmentation in augmentations:
        norm_train_normalized = np.concatenate([base, utils.augment2(base, augmentation)])

    return norm_train_normalized, mal_train_normalized, {'real-test': (norm_test_normalized, mal_test_normalized)}


if __name__ == '__main__':
    pass
    # load_mit_bih('C:\workarea\mit-bih\\')
    # _ = load_sliced("/home/robin/Documents/lbnl/crd/RealDatasets/ANNOTATIONS/", **{'preprocessing': {'time-steps': 60}})
    # _ = load_sliced()
