import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wfdb
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import utils
from data.mit_bih import load_normal


def p_at_k(vae, downstream_model, task, **config):
    window_size = config['preprocessing']['time-steps']
    data_location = config['data']['location']
    anot_file = data_location + str(803) + '/' + str(803)
    anot = wfdb.rdann(anot_file, 'atr')
    mit_set = load_normal(config['data']['location'], nr=803)[0]
    eval_data = MinMaxScaler().fit_transform(mit_set.reshape(-1, 1))
    nr_sequences = math.floor(eval_data.shape[0] / window_size)
    eval_idxs = np.arange(eval_data.shape[0]).reshape(nr_sequences, window_size, 1)
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
    conf_matrix = confusion_matrix(np.ones(perf_relevant.shape[0]), preds_at_k)
    prec = precision_score(y_at_k, preds_at_k)
    recall = recall_score(y_at_k, preds_at_k)
    f1 = f1_score(y_at_k, preds_at_k)
    acc = accuracy_score(y_at_k, preds_at_k)
    print('p@k prec:' + str(prec))
    print('p@k rec:' + str(recall))
    print('p@k f1: ' + str(f1))
    print('p@k acc: ' + str(acc))
    print(str(conf_matrix))
    wandb.run.summary["Precision@k_task_" + task + "_scenario_803"] = prec
    wandb.run.summary["Recall@k_task_" + task + "_scenario_803"] = recall
    wandb.run.summary["F1@k_task_" + task + "_scenario_803"] = f1
    wandb.run.summary["Accuracy@k_task_" + task + "_scenario_803"] = acc

    log = pd.DataFrame(columns=['score_norm', 'score_mal', 'prediction', 'label'], data=y_and_mal_score)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/803-p@k-raw_data.csv')


def eval(test_data, vae, downstream_model, scenario, file_prefix, run_id, task, **config):
    test_norm, test_bot = test_data
    p_at_k(vae, downstream_model, task, **config)
    mit_set = load_normal(config['data']['location'], nr=803)[0]
    eval_data = MinMaxScaler().fit_transform(mit_set.reshape(-1, 1))
    window_size = test_norm.shape[1]
    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)

    preds_norm = []
    preds_mal = []
    recon_errs = []
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(eval_data.shape[0] - window_size - 1)[:, None]
    sl_windows = eval_data[indexer]
    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, sl_windows, axis=(1, 2))
    preds = score_mal > score_norm
    mal_idx = (score_mal > score_norm).nonzero()[0]

    start = round(12 * 128)
    end = round(18 * 128)
    min_in_slice = np.min(mit_set[window_size: eval_data.shape[0]][start:end])
    max_in_slice = np.max(mit_set[window_size: eval_data.shape[0]][start:end])
    fill_int = mal_idx[np.isin(mal_idx, np.arange(start, end))]
    fill_int = np.split(fill_int, np.where(np.diff(fill_int) != 1)[0] + 1)
    _, ax = plt.subplots(2, 1, figsize=(17, 5))
    ax[0].plot(list(range(0, eval_data.shape[0]))[start:end], mit_set[window_size: eval_data.shape[0]][start:end],
               label='ecg')
    for x in fill_int:
        ax[0].fill_between(x, min_in_slice, max_in_slice, alpha=0.2, color='red')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], score_norm[start:end], color='green',
               label='likelihood - normal')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], score_mal[start:end], color='red',
               label='likelihood - anomaly')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], preds[start:end], color='blue',
               label='is anomaly')
    ax[0].set_xlabel('frame')
    ax[0].set_ylabel('mm')
    ax[1].set_xlabel('frame')
    ax[1].set_ylabel('likelihood')
    ax[0].legend()
    ax[1].legend()
    plt.savefig(config['run-dir'] + '/downstream-task/' + task + '/sliding-window-eval-803.png')
    plt.show()

    log = pd.DataFrame(columns=['recon-err', 'score_norm', 'score_mal'],
                       data=np.stack([recon_err, score_norm, score_mal], axis=1))
    log_name = file_prefix + '-sliding-windows-scores-' + str(scenario)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/' + log_name + '.csv')
