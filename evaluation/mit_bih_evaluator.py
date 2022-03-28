import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import utils
from data.mit_bih import load_normal


def p_at_k(vae, downstream_model, task, **config):
    logger = utils.get_logger(**config)
    window_size = config['preprocessing']['time-steps']
    data_location = config['data']['location']
    anot_file = data_location + str(803) + '/' + str(803)
    anot = wfdb.rdann(anot_file, 'atr')
    mit_set = load_normal(config['data']['location'], nr=803)[0]
    eval_data = MinMaxScaler().fit_transform(mit_set.reshape(-1, 1))
    nr_sequences = math.floor(eval_data.shape[0] / window_size)
    eval_idxs = np.arange(nr_sequences * window_size).reshape(nr_sequences, window_size, 1)
    eval_data = eval_data[:window_size * nr_sequences]
    eval_data = eval_data.reshape(nr_sequences, window_size, 1)
    mal_idxs = np.array(anot.sample)[(np.array(anot.symbol) != 'N').nonzero()[0]]
    y = np.isin(np.arange(eval_data.shape[0]), np.isin(eval_idxs, mal_idxs).astype(int).nonzero()[0]).astype(int)

    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, eval_data, axis=(1, 2))
    preds_test = score_mal > score_norm
    preds_test = preds_test.astype(int).reshape(y.shape)
    y_and_mal_score = np.stack([score_norm, score_mal, preds_test, y], 1)
    y_mal_sorted = y_and_mal_score[y_and_mal_score[:, 1].argsort()[::-1]]

    perf_relevant = y_mal_sorted[:mal_idxs.shape[0]]
    y_at_k = perf_relevant[:, 3]
    preds_at_k = perf_relevant[:, 2]
    prec = precision_score(y_at_k, preds_at_k)
    recall = recall_score(y_at_k, preds_at_k)
    f1 = f1_score(y_at_k, preds_at_k)
    acc = accuracy_score(y_at_k, preds_at_k)
    context = 'eval-' + task
    logger.info('', extra={'summary': {'context': context, 'entity': 'scen-803', 'item': 'precsion@k', 'data': prec}})
    logger.info('', extra={'summary': {'context': context, 'entity': 'scen-803', 'item': 'recall@k', 'data': recall}})
    logger.info('', extra={'summary': {'context': context, 'entity': 'scen-803', 'item': 'f1@k', 'data': f1}})
    logger.info('', extra={'summary': {'context': context, 'entity': 'scen-803', 'item': 'accuracy@k', 'data': acc}})

    log = pd.DataFrame(columns=['score_norm', 'score_mal', 'prediction', 'label'], data=y_and_mal_score)
    logger.info('', extra={'data': {'context': 'eval', 'entity': 'scencario-803', 'item': 'raw-data', 'data': log}})


def eval_all(test_data, vae, downstream_model, scenario, task, **config):
    logger = utils.get_logger(**config)
    context = 'eval-' + task
    entity = 'scenario-' + scenario
    window_size = config['preprocessing']['time-steps']
    mit_set = load_normal(config['data']['location'], nr=803)[0]
    eval_data = MinMaxScaler().fit_transform(mit_set.reshape(-1, 1))
    test_data = eval_data[0:math.floor(eval_data.shape[0] / window_size) * window_size]
    test_data = test_data.reshape(-1, window_size, 1)

    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, test_data, axis=(1, 2))
    preds = score_mal > score_norm

    start = round(12 * 128)
    end = round(18 * 128)
    min_in_slice = np.min(mit_set[window_size: eval_data.shape[0]][start:end])
    max_in_slice = np.max(mit_set[window_size: eval_data.shape[0]][start:end])

    # fill_int = np.split(fill_int, np.where(np.diff(fill_int) != 1)[0] + 1)
    score_norm_rep = np.repeat(score_norm, window_size)
    score_mal_rep = np.repeat(score_mal, window_size)
    preds_rep = np.repeat(preds, window_size)
    xx = np.arange(eval_data.shape[0])
    fill = xx[start:end][np.isin(xx[start:end], preds_rep.nonzero()).nonzero()]
    fill_idx = np.split(fill, np.where(np.diff(fill) != 1)[0] + 1)

    _, ax = plt.subplots(2, 1, figsize=(17, 5))
    ax[0].plot(xx[start:end], mit_set[window_size: eval_data.shape[0]][start:end], label='ecg')
    for x in fill_idx:
        ax[0].fill_between(x, -1, 2, alpha=0.2, color='red')
    ax[1].step(xx[start:end], score_norm_rep[start:end], color='green', label='KDE value - normal')
    ax[1].step(xx[start:end], score_mal_rep[start:end], color='red', label='KDE value - anomaly')
    ax[1].step(xx[start:end], preds_rep[start:end], color='blue', label='is anomaly')
    ax[0].set_xlabel('frame')
    ax[0].set_ylabel('mm')
    ax[1].set_xlabel('frame')
    ax[1].set_ylabel('likelihood')
    ax[0].legend()
    ax[1].legend()
    task_dir = config['run-dir'] + '/downstream-task/' + task
    Path(task_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(task_dir + '/sliding-window-eval-803.png')
    plt.show()

    log = pd.DataFrame(columns=['recon-err', 'score_norm', 'score_mal'],
                       data=np.stack([recon_err, score_norm, score_mal], axis=1))
    sl_item = 'sliding-windows-scores'
    logger.info('', extra={'data': {'context': context, 'entity': entity, 'item': sl_item, 'data': log}})


def eval(test_data, vae, downstream_model, scenario, task, **config):
    test_norm, test_bot = test_data
    logger = utils.get_logger(**config)
    context = 'eval-' + task
    entity = 'scenario-' + scenario
    eval_all(test_data, vae, downstream_model, scenario, task, **config)
    p_at_k(vae, downstream_model, task, **config)
    mit_set = load_normal(config['data']['location'], nr=803)[0]
    eval_data = MinMaxScaler().fit_transform(mit_set.reshape(-1, 1))
    window_size = test_norm.shape[1]

    test_d = np.concatenate([test_norm, test_bot], axis=0)
    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, test_d, axis=(1, 2))
    preds_test = score_mal > score_norm
    preds_test = preds_test.astype(int).reshape(y.shape)
    y_and_mal_score = np.stack([score_norm, score_mal, preds_test, y], 1)

    prec = precision_score(preds_test, y)
    recall = recall_score(preds_test, y)
    f1 = f1_score(preds_test, y)
    acc = accuracy_score(preds_test, y)
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'precsion', 'data': prec}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'recall', 'data': recall}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'f1', 'data': f1}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'accuracy', 'data': acc}})

    log = pd.DataFrame(columns=['score_norm', 'score_mal', 'prediction', 'label'], data=y_and_mal_score)
    logger.info('', extra={'data': {'context': context, 'entity': entity, 'item': 'raw-data', 'data': log}})
