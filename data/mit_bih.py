import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
import utils
import matplotlib.pyplot as plt


def preprocess(data_ar, is_training, time_steps=5, **kwargs):
    augmentation = []
    if 'preprocessing' in kwargs:
        augmentation = kwargs['preprocessing']['augmentation'] if 'augmentation' in kwargs['preprocessing'] else []
    p_data = np.array([rec[0: math.floor(rec.shape[0] / time_steps) * time_steps] for rec in data_ar])
    p_data = np.concatenate([rec.reshape(math.floor(rec.shape[0] / time_steps), time_steps) for rec in p_data])
    p_data = MinMaxScaler().fit_transform(p_data)
    p_data = p_data.reshape(p_data.shape[0], p_data.shape[1], 1)
    if is_training:
        base = p_data.copy()
        for augment in augmentation:
            p_data = np.concatenate([p_data, utils.augment2(base, augment)])
    return p_data


def load(cfg):
    data_location = cfg['data']['location']
    train_sets = cfg['data']['train-sets']
    test_groups = cfg['data']['test-groups']
    t_steps = cfg['preprocessing']['time-steps']
    # _ = load_sliced(data_location, nr=train_sets[0], **cfg)
    train_norm, train_mal = load_multiple2(data_location, train_sets, **cfg)
    train_norm, test_norm, _, _ = train_test_split(train_norm, np.zeros(train_norm.shape[0]), test_size=0.2)
    train_mal, test_mal, _, _ = train_test_split(train_mal, np.zeros(train_mal.shape[0]), test_size=0.2)
    train_norm = preprocess(train_norm, True, time_steps=t_steps, **cfg)
    train_mal = preprocess(train_mal, True, time_steps=t_steps, **cfg)

    test_norm = preprocess(test_norm, False, time_steps=t_steps, **cfg)
    test_mal = preprocess(test_mal, False, time_steps=t_steps, **cfg)

    test_sets = {'real-test': (test_norm, test_mal)}
    for group in test_groups:
        test_norm, test_mal = load_multiple2(data_location, test_groups[group], **cfg)
        test_norm = preprocess(test_norm, False, time_steps=t_steps)
        test_mal = preprocess(test_mal, False, time_steps=t_steps)
        test_sets[group] = (test_norm, test_mal)

    return train_norm, train_mal, test_sets


def load_multiple(data_location, sets=None, **kwargs):
    sets = np.concatenate([load_mit_bih(data_location, nr=set_nr, **kwargs) for set_nr in sets])
    norm = np.concatenate(sets[0::2])
    mal = np.concatenate(sets[1::2])

    return norm, mal


def load_normal(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    return sigs


def load_multiple2(data_location, sets=None, **kwargs):
    sets = np.concatenate([load_sliced(data_location, nr=set_nr, **kwargs) for set_nr in sets])
    norm = np.concatenate(sets[0::2])
    mal = np.concatenate(sets[1::2])

    return norm, mal


def load_sliced(data_location, nr=805, **kwargs):
    window_size = kwargs['preprocessing']['time-steps']
    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])[0]
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(sigs.shape[0] - window_size - 1)[:, None]
    mal_samples = sigs[indexer[np.isin(indexer, np.array(anot.sample)[np.array(anot.symbol) != 'N']).nonzero()[0]]]
    norm_samples = sigs[indexer[np.isin(indexer, np.array(anot.sample)[np.array(anot.symbol) == 'N']).nonzero()[0]]]
    return norm_samples, mal_samples


def load_mit_bih(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr) + '/' + str(nr)
    data_file = data_location + str(nr) + '/' + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    norm = []
    mal = []
    prev = 0
    length = round(anot.fs / 2)
    for i in range(0, len(anot.symbol) - 1):
        if anot.symbol[i] != 'N':
            mal.append(sigs[0][anot.sample[i]:anot.sample[i] + length])
            norm.append(sigs[0][anot.sample[i] + length:anot.sample[i + 1]])

            mal.append(sigs[1][anot.sample[i]:anot.sample[i] + length])
            norm.append(sigs[1][anot.sample[i] + length + length:anot.sample[i + 1]])

        else:
            norm.append(sigs[0][anot.sample[i]:anot.sample[i + 1]])
            norm.append(sigs[1][anot.sample[i]:anot.sample[i + 1]])
        prev = anot.sample[i]
    norm = np.array(norm)
    mal = np.array(mal)
    return np.array([norm, mal])


def load_mit_bih2(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr)
    data_file = data_location + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    mal_bound = round(anot.fs / 10)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    mal_idx = np.argwhere(np.array(anot.symbol) != 'N')
    mal_ranges = np.array([np.array([anot.sample[mal][0] - mal_bound, (
            anot.sample[mal] + np.round((anot.sample[mal_idx[idx] + 1][0] - anot.sample[mal]) / 2)).astype(int)[0]])
                           for idx, mal in enumerate(mal_idx)])
    mal_ranges_values = np.array([np.array(list(range(interval[0], interval[1]))) for interval in mal_ranges])
    mal_ex = np.array([sigs[0][r] for r in mal_ranges_values])
    xx = range(0, mal_ex[0].shape[0])
    plt.plot(xx, mal_ex[0])
    plt.show()


def fetch_sliced(cfg):
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
    norm_train = sigs[indexer[np.isin(indexer, norm_idxs_train).nonzero()[0]]]
    norm_test = sigs[indexer[np.isin(indexer, norm_idxs_test).nonzero()[0]]]
    mal_train = sigs[indexer[np.isin(indexer, mal_idx_train).nonzero()[0]]]
    mal_test = sigs[indexer[np.isin(indexer, mal_idxs_test).nonzero()[0]]]

    normalized_train = MinMaxScaler().fit_transform(np.concatenate([norm_train, mal_train]))
    norm_train_normalized = normalized_train[0:norm_train.shape[0]]
    norm_train_normalized = norm_train_normalized.reshape(norm_train_normalized.shape[0], norm_train_normalized.shape[1], 1)
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
    # load_mit_bih('C:\workarea\mit-bih\\')
    _ = load_sliced("/home/robin/Documents/lbnl/crd/RealDatasets/ANNOTATIONS/", **{'preprocessing': {'time-steps': 60}})
    # _ = load_sliced()
    preprocess('norm')
