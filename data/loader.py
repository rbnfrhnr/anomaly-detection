import os
import pandas as pd
import utils
from data import preprocessing


def fetch_set(ctu_set, filepath):
    return pd.read_csv(filepath + str(ctu_set) + '/' + utils.files[ctu_set - 1])


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

    if os.path.isfile(train_cache_file) and not override_cache:
        train = pd.read_csv(train_cache_file, index_col=0)
    else:
        train = pd.concat([fetch_set(ctu_set, data_location) for ctu_set in train_sets])
        train = preprocessing.preprocess_stat_2_CTU(train, **preprcs_params)
        utils.save(train, train_cache_file)

    test_group_data = {}
    for test_group in test_groups:
        cache_file = cache_location + (test_group + test_group_cache_suffix) + '.csv'
        if os.path.isfile(cache_file) and not override_cache:
            test_group_data[test_group] = pd.read_csv(cache_file, index_col=0)
        else:
            test_group_data[test_group] = pd.concat([fetch_set(ctu_set) for ctu_set in test_groups[test_group]])
            test_group_data[test_group] = preprocessing.preprocess_stat_2_CTU(test_group_data[test_group],
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
    return train, test_group_data