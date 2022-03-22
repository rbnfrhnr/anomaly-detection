import numpy as np
import wandb
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
import utils
import pandas as pd


def test_eval(test_data, vae, downstream_model, scenario, task, **config):
    logger = utils.get_logger(**config)
    logger.info('test eval for scenario ' + str(scenario))
    test_norm, test_bot = test_data

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = utils.predict(vae, downstream_model, np.concatenate([test_norm, test_bot]), axis=(1, 2))
    preds = preds.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds)
    prec = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)

    entity = 'scenario-' + str(scenario)
    context = 'eval-' + task
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'precsion', 'data': prec}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'recall', 'data': recall}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'f1', 'data': f1}})
    logger.info('', extra={'summary': {'context': context, 'entity': entity, 'item': 'accuracy', 'data': acc}})

    log_name = 'reconstruction-error'
    log = pd.DataFrame(columns=['recon-err', 'prediction', 'label'], data=np.stack([recon_err, preds, y], axis=1))
    logger.info('', extra={'data': {'context': 'eval', 'entity': log_name, 'item': 'raw-data', 'data': log}})
