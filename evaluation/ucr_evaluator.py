import utils
import numpy as np
import pandas as pd


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
    max_anomaly = np.argwhere(score_norm == score_norm.min())

    found = y_2[max_anomaly][0][0]
    entity = 'scenario-' + str(scenario)
    context = 'eval-' + task
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found', 'data': found}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'predicted-index-rel-to-test',
                    'data': max_anomaly[0][0] * t_steps}})
    logger.info('', extra={
        'summary': {'context': context, 'entity': entity, 'item': 'binary-anomaly-found-int', 'data': int(found)}})

    log_name = 'reconstruction-error'
    log = pd.DataFrame(columns=['recon-err', 'score-norm', 'score-mal', 'label'],
                       data=np.stack([recon_err, score_norm, score_mal, y_2], axis=1))
    logger.info('', extra={'data': {'context': 'eval', 'entity': log_name, 'item': 'raw-data', 'data': log}})
