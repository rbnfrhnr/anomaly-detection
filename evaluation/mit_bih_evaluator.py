import math
import wandb
import matplotlib.pyplot as plt
import numpy as np
import utils
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from data.mit_bih import load_normal
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def eval(test_data, vae, downstream_model, scenario, file_prefix, run_id, task, **config):
    test_norm, test_bot = test_data
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
    ax[0].plot(list(range(0, eval_data.shape[0]))[start:end], mit_set[window_size: eval_data.shape[0]][start:end])
    for x in fill_int:
        ax[0].fill_between(x, min_in_slice, max_in_slice, alpha=0.2, color='red')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], score_norm[start:end], color='green')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], score_mal[start:end], color='red')
    ax[1].plot(list(range(0, eval_data.shape[0] - window_size - 1))[start:end], preds[start:end], color='blue')
    plt.savefig(config['run-dir'] + '/downstream-task/' + task + '/sliding-window-eval-803.png')
    plt.show()

    log = pd.DataFrame(columns=['recon-err', 'score_norm', 'score_mal'],
                       data=np.stack([recon_err, score_norm, score_mal], axis=1))
    log_name = file_prefix + '-sliding-windows-scores-' + str(scenario)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/' + log_name + '.csv')

    preds_test, recon_err = utils.predict(vae, downstream_model, np.concatenate([test_norm, test_bot]), axis=(1, 2))
    preds_test = preds_test.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds_test)
    prec = precision_score(y, preds_test)
    recall = recall_score(y, preds_test)
    f1 = f1_score(y, preds_test)
    acc = accuracy_score(y, preds_test)
    print('#normal: ' + str(test_norm.shape[0]) + ' / #bot: ' + str(test_bot.shape[0]))
    print('prec:' + str(prec))
    print('rec:' + str(recall))
    print('f1: ' + str(f1))
    print('acc: ' + str(acc))
    print(str(conf_matrix))

    wandb.run.summary["Precision_task_" + task + "_scenario_" + str(scenario)] = prec
    wandb.run.summary["Recall_task_" + task + "_scenario_" + str(scenario)] = recall
    wandb.run.summary["F1_task_" + task + "_scenario_" + str(scenario)] = f1
    wandb.run.summary["Accuracy_task_" + task + "_scenario_" + str(scenario)] = acc
    wandb.sklearn.plot_confusion_matrix(y, preds_test, ['normal', 'botnet'])
    hist_data = np.stack([recon_err, y], axis=1)

    hist_norm = hist_data[0:test_norm.shape[0]]
    if hist_norm.shape[0] > 5000:
        # hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000), :]
        hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000)]

    hist_mal = hist_data[test_norm.shape[0]:, ]
    if hist_mal.shape[0] > 5000:
        # hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000), :]
        hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000)]

    table_name = 'reconstruction-error-test-taks-' + task + '_scenario-' + str(scenario)
    wandb.log(
        {table_name: wandb.Table(
            data=np.concatenate([hist_norm, hist_mal]).tolist(),
            columns=['reconstruction-error', 'label'])})
    log = pd.DataFrame(columns=['recon-err', 'prediction', 'label'], data=np.stack([recon_err, preds_test, y], axis=1))
    log_name = file_prefix + '-reconstruction-error-' + str(scenario)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/' + log_name + '.csv')
