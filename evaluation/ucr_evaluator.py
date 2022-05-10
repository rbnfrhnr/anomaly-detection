import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(test_data, vae, downstream_model, scenario, task, **config):
    logger = utils.get_logger(**config)
    t_steps = config['preprocessing']['time-steps']
    X, y = test_data
    X = X.reshape(-1, t_steps, 1)
    y = y.reshape(-1, t_steps, 1)
    y_2 = y.sum(axis=1)
    y_2 = y_2 >= 1
    y_2 = y_2.reshape(-1)

    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, X, axis=(1, 2))
    current_min = 999
    curent_idx = 0
    for idx in range(0, score_norm.shape[0] - 2):
        mean = np.mean(score_norm[idx:idx + 2])
        if mean < current_min:
            current_min = mean
            curent_idx = np.arange(idx, idx + 2)

    # found = y_2[curent_idx]
    found = np.any(y_2[curent_idx] == True)
    max_anomaly = current_min
    # max_anomaly = np.argwhere(score_norm == score_norm.min())

    # found = y_2[max_anomaly][0][0]
    entity = 'scenario-' + str(scenario)
    context = 'eval-' + task
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found', 'data': found}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'predicted-index-rel-to-test',
                    'data': curent_idx[0] * t_steps}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found-int', 'data': int(found)}})

    log_name = 'reconstruction-error'
    log = pd.DataFrame(columns=['recon-err', 'score-norm', 'score-mal', 'label'],
                       data=np.stack([recon_err, score_norm, score_mal, y_2], axis=1))
    logger.info('', extra={'data': {'context': 'eval', 'entity': log_name, 'item': 'raw-data', 'data': log}})

    # png_file_name = config['run-dir'] + '/' + context + '-' + entity + '-actual-vs-predict.png'
    # padding = 3
    # real_idx = np.argwhere(y_2 == True).reshape(-1)
    # anomaly_scores = np.repeat(score_norm, t_steps)
    # start_real = real_idx[0] - padding
    # end_real = real_idx[-1] + padding
    # start_pred = curent_idx[0] - padding
    # end_pred = curent_idx[-1] + padding
    # x_real = np.arange(start_real * t_steps, end_real * t_steps)
    # x_pred = np.arange(start_pred * t_steps, end_pred * t_steps)
    # _, ax = plt.subplots(3, 1, figsize=(17, 10))
    # ax[0].plot(x_real, X[start_real:end_real].reshape(-1), label='test data')
    # ax[0].fill_between((real_idx * t_steps) + ([0] * (real_idx.shape[0] - 1) + [t_steps]), [1] * real_idx.shape[0],
    #                    alpha=0.2)
    # ax[0].plot(x_real, anomaly_scores[start_real * t_steps:end_real * t_steps], label='normality score')
    # ax[0].legend()
    # ax[1].plot(x_pred, X[start_pred:end_pred].reshape(-1), label='test data')
    # ax[1].fill_between((curent_idx * t_steps) + ([0] * (curent_idx.shape[0] - 1) + [t_steps]),
    #                    [1] * curent_idx.shape[0], alpha=0.2)
    # ax[1].plot(x_pred, anomaly_scores[start_pred * t_steps:end_pred * t_steps], label='normality score')
    # ax[1].legend()
    # ax[2].plot(np.arange(X.shape[0] * t_steps), X.reshape(-1), label='test data')
    # ax[2].plot(np.arange(X.shape[0] * t_steps), anomaly_scores, alpha=0.4, label='normality score')
    # ax[2].fill_between((real_idx * t_steps) + ([0] * (real_idx.shape[0] - 1) + [t_steps]), [1] * real_idx.shape[0],
    #                    alpha=0.3, color='red',
    #                    label='real anomaly')
    # ax[2].fill_between((curent_idx * t_steps) + ([0] * (curent_idx.shape[0] - 1) + [t_steps]), [1] * real_idx.shape[0],
    #                    alpha=0.3, color='grey',
    #                    label='predicted anomaly')
    # ax[2].legend()
    # plt.savefig(png_file_name)
    # plt.show()
