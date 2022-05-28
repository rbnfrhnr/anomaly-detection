import os

import numpy as np
import pandas as pd

import utils
from data import preprocessing


def fetch_set(ctu_set, filepath):
    return pd.read_csv(filepath + str(ctu_set) + '/' + utils.files[ctu_set - 1])


def get_preprocess_method(preprocess_type='medium'):
    if preprocess_type == 'medium':
        return lambda data, **kwargs: preprocessing.preprocess_stat_2_CTU(data, **kwargs)
    if preprocess_type == 'large':
        return lambda data, **kwargs: preprocessing.preprocess_stat_CTU(data, **kwargs)
    if preprocess_type == 'connection-only':
        return lambda data, **kwargs: preprocessing.preprocess_CTU(data, **kwargs)
    if preprocess_type == 'stat-3':
        return lambda data, **kwargs: preprocessing.preprocess_stat_3(data, **kwargs)
    if preprocess_type == 'medium-by-host':
        return lambda data, **kwargs: preprocessing.preprocess_ctu_medium_by_host(data, **kwargs)


def load(config):
    data_location = config['data']['location']
    cache_location = config['preprocessing']['cache']['location']
    override_cache = config['preprocessing']['cache']['override']
    train_sets = config['data']['train-ctu-sets']
    test_groups = config['data']['test-groups']
    train_cache_name = 'train-' + config['preprocessing']['type']
    test_group_cache_suffix = '-test-' + config['preprocessing']['type']
    train_cache_file = cache_location + train_cache_name + '.csv'
    preprcs_params = config['preprocessing']['params']
    preprocess_method = get_preprocess_method(config['preprocessing']['type'])
    augmentations = []
    base_augmentations = []
    if 'preprocessing' in config:
        augmentations = config['preprocessing']['augmentation'] if 'augmentation' in config['preprocessing'] else []
        base_augmentations = config['preprocessing']['base-augmentation'] if 'base-augmentation' in config[
            'preprocessing'] else []

    if os.path.isfile(train_cache_file) and not override_cache:
        train = pd.read_csv(train_cache_file, index_col=0)
    else:
        train = pd.concat([fetch_set(ctu_set, data_location) for ctu_set in train_sets])
        train = preprocess_method(train, **preprcs_params)
        utils.save(train, train_cache_file)

    test_group_data = {}
    for test_group in test_groups:
        cache_file = cache_location + (test_group + test_group_cache_suffix) + '.csv'
        if os.path.isfile(cache_file) and not override_cache:
            test_group_data[test_group] = pd.read_csv(cache_file, index_col=0)
        else:
            test_group_data[test_group] = pd.concat(
                [fetch_set(ctu_set, data_location) for ctu_set in test_groups[test_group]])
            test_group_data[test_group] = preprocess_method(test_group_data[test_group],
                                                            **preprcs_params)
            utils.save(test_group_data[test_group], cache_file)

    d = [test_group_data[d].columns.values for d in test_group_data.keys()]
    d.append(train.columns.values)
    common_attributes = set.intersection(*map(set, d))
    train = train.drop(columns=[feat for feat in train.columns if feat not in common_attributes])
    test_group_data = {
        group: test_group_data[group].drop(
            columns=[feat for feat in test_group_data[group].columns if feat not in common_attributes])
        for group in test_group_data.keys()}

    for test_group in test_groups:
        test_norm, test_mal = utils.split_mal_norm(test_group_data[test_group])
        test_norm = test_norm.drop(columns=['class']).values
        test_mal = test_mal.drop(columns=['class']).values
        test_norm = utils.reshape_for_rnn(test_norm, 5)
        test_mal = utils.reshape_for_rnn(test_mal, 5)
        test_group_data[test_group] = (test_norm, test_mal)

    train_norm, train_mal = utils.split_mal_norm(train)
    train_norm = train_norm.drop(columns=['class']).values
    train_norm = utils.reshape_for_rnn(train_norm)
    train_mal = train_mal.drop(columns=['class']).values
    train_mal = utils.reshape_for_rnn(train_mal)
    base_norm = train_norm.copy()
    base_mal = train_mal.copy()
    for base_augmentation in base_augmentations:
        base_norm = utils.base_augmentation(base_norm, base_augmentation)
        base_mal = utils.base_augmentation(base_mal, base_augmentation)
    train_norm = base_norm
    train_mal = base_mal
    for augmentation in augmentations:
        train_norm = np.concatenate([base_norm, utils.augment(base_norm, augmentation)])
        # train_mal = np.concatenate([base_mal, utils.augment(base_mal, augmentation)])

    return train_norm, train_mal, test_group_data


def load_by_host(config):
    data_location = config['data']['location']
    cache_location = config['preprocessing']['cache']['location']
    override_cache = config['preprocessing']['cache']['override']
    train_sets = config['data']['train-ctu-sets']
    test_groups = config['data']['test-groups']
    train_cache_name = 'train-' + config['preprocessing']['type']
    test_group_cache_suffix = '-test-' + config['preprocessing']['type']
    train_cache_file = cache_location + train_cache_name + '.csv'
    preprcs_params = config['preprocessing']['params']
    preprocess_method = get_preprocess_method(config['preprocessing']['type'])
    augmentations = []
    base_augmentations = []
    if 'preprocessing' in config:
        augmentations = config['preprocessing']['augmentation'] if 'augmentation' in config['preprocessing'] else []
        base_augmentations = config['preprocessing']['base-augmentation'] if 'base-augmentation' in config[
            'preprocessing'] else []

    train = pd.concat([fetch_set(ctu_set, data_location) for ctu_set in train_sets])
    train_norm, train_mal = preprocess_method(train, **preprcs_params)

    test_group_data = {}
    for test_group in test_groups:
        cache_file = cache_location + (test_group + test_group_cache_suffix) + '.csv'
        test_group_data[test_group] = pd.concat(
            [fetch_set(ctu_set, data_location) for ctu_set in test_groups[test_group]])
        test_group_data[test_group] = preprocess_method(test_group_data[test_group],
                                                        **preprcs_params)

    base_norm = train_norm.copy()
    base_mal = train_mal.copy()
    for base_augmentation in base_augmentations:
        base_norm = utils.base_augmentation(base_norm, base_augmentation)
        base_mal = utils.base_augmentation(base_mal, base_augmentation)
    train_norm = base_norm
    train_mal = base_mal
    for augmentation in augmentations:
        train_norm = np.concatenate([base_norm, utils.augment(base_norm, augmentation)])
        # train_mal = np.concatenate([base_mal, utils.augment(base_mal, augmentation)])

    return train_norm, train_mal, test_group_data
