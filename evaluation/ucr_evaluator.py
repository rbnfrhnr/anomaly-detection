import numpy as np
import pandas as pd

import utils


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


def evaluate2(test_data, vae, downstream_model, scenario, task, **config):
    logger = utils.get_logger(**config)
    window_size = config['preprocessing']['time-steps']
    X_original, y = test_data
    X = X_original
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(X.shape[0] - window_size - 1)[:, None]
    X_sliding = X[indexer]
    X_sliding = X_sliding.reshape(-1, window_size, 1)
    score_norm, score_mal, recon_err = utils.predict2(vae, downstream_model, X_sliding, axis=(1, 2))

    idx = np.arange(X_sliding.shape[0])

    dim = score_mal.shape[0]
    matrix = np.zeros(shape=(dim, dim))
    for i in range(dim):
        matrix[i][i:min(i + window_size, dim)] = np.repeat(score_norm[i], min(window_size, dim - i))
    mean_window = np.mean(matrix, axis=0, where=(matrix > 0))
    mean_window[np.isnan(mean_window)] = np.inf
    min_idx = 0
    min_val = np.inf
    for i in range(0, mean_window.shape[0]):
        current = np.mean(mean_window[i: i + window_size])
        if current < min_val:
            min_val = current
            min_idx = i

    predicted_idx = np.arange(min_idx, min_idx + window_size)
    actual_idx = (y == 1).nonzero()[0]

    found = np.in1d(predicted_idx, actual_idx).nonzero()[0].shape[0] > 0

    entity = 'scenario-' + str(scenario)
    context = 'eval-' + task
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found',
                    'data': found}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found-int',
                    'data': int(found)}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'anomaly-starting-index',
                    'data': min_idx}})

    log_name = 'reconstruction-error-mean'
    log = pd.DataFrame(columns=['mean-score-norm', 'label'], data=np.stack([mean_window, y[:-window_size - 1]], axis=1))
    logger.info('', extra={'data': {'context': task + '-eval', 'entity': log_name, 'item': 'raw-data', 'data': log}})

    log_name = 'reconstruction-error'
    log = pd.DataFrame(columns=['recon-err', 'score-norm', 'score-mal', 'label'],
                       data=np.stack([recon_err, score_norm, score_mal, y[:-window_size - 1]], axis=1))
    logger.info('', extra={'data': {'context': task + '-eval', 'entity': log_name, 'item': 'raw-data', 'data': log}})
