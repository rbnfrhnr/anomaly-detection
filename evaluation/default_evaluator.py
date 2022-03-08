import numpy as np
import wandb
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
import utils
import pandas as pd


def test_eval(test_data, vae, downstream_model, scenario, file_prefix, run_id, task, **config):
    print('test eval for scenario ' + str(scenario))
    test_norm, test_bot = test_data

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = utils.predict(vae, downstream_model, np.concatenate([test_norm, test_bot]), axis=(1, 2))
    preds = preds.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds)
    prec = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
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
    wandb.sklearn.plot_confusion_matrix(y, preds, ['normal', 'botnet'])
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
    log = pd.DataFrame(columns=['recon-err', 'prediction', 'label'], data=np.stack([recon_err, preds, y], axis=1))
    log_name = file_prefix + '-reconstruction-error-' + str(scenario)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/' + log_name + '.csv')
