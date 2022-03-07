import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
import utils


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
    train_norm, train_mal = load_multiple(data_location, train_sets, **cfg)
    train_norm, test_norm, _, _ = train_test_split(train_norm, np.zeros(train_norm.shape[0]), test_size=0.2)
    train_mal, test_mal, _, _ = train_test_split(train_mal, np.zeros(train_mal.shape[0]), test_size=0.2)
    train_norm = preprocess(train_norm, True, time_steps=t_steps, **cfg)
    train_mal = preprocess(train_mal, True, time_steps=t_steps, **cfg)

    test_norm = preprocess(test_norm, False, time_steps=t_steps, **cfg)
    test_mal = preprocess(test_mal, False, time_steps=t_steps, **cfg)

    test_sets = {'real-test': (test_norm, test_mal)}
    for group in test_groups:
        test_norm, test_mal = load_multiple(data_location, test_groups[group])
        test_norm = preprocess(test_norm, False, time_steps=t_steps)
        test_mal = preprocess(test_mal, False, time_steps=t_steps)
        test_sets[group] = (test_norm, test_mal)

    return train_norm, train_mal, test_sets


def load_multiple(data_location, sets=None, **kwargs):
    sets = np.concatenate([load_mit_bih(data_location, nr=set_nr, **kwargs) for set_nr in sets])
    norm = np.concatenate(sets[0::2])
    mal = np.concatenate(sets[1::2])

    return norm, mal


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


if __name__ == '__main__':
    # load_mit_bih('C:\workarea\mit-bih\\')
    norm, mal = load_multiple('C:\workarea\mit-bih\\')
    preprocess(norm)
