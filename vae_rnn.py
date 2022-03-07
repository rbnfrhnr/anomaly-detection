import math
import sys
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from tensorflow import keras
from keras import backend
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from data import loader_factory
from downstream import downstream_factory
import tensorflow as tf

import utils
import wandb
from model.rvae import RVAE

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = utils.read_cfg(config_file)
    config = utils.setup_run(config)
    use_gpu = config['use-gpu']

    print(backend._get_available_gpus())
    if use_gpu:
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        backend.set_session(sess)

    data_set = config['data']['data-set']
    downstream_tasks = config['downstream']['tasks']
    wandb_group_name = config['experiment-name']
    wandb_project = config['logging']['wandb']['project']
    wandb_entity = config['logging']['wandb']['entity']
    wandb.init(project=wandb_project, entity=wandb_entity,
               group=str(wandb_group_name))

    run_id = config['run-id']
    latent_dim = config['autoencoder']['latent-dim']
    batch_size = config['autoencoder']['batch-size']
    epochs = config['autoencoder']['epochs']

    train_norm, train_bot, test_scenarios = loader_factory.get_loader(data_set)(config)
    test = test_scenarios['all']


    test_norm, test_bot = test

    feature_dim = train_norm.shape
    model = utils.create_model(config, feature_dim=feature_dim)
    model.fit(train_norm, epochs=epochs, batch_size=batch_size, callbacks=[WandbCallback()])

    err_normal_train, mu_normal_train, log_sig_normal_train = utils.pred(model, train_norm, axis=(1, 2))
    err_bot_train, mu_bot_train, log_sig_bot_train = utils.pred(model, train_bot, axis=(1, 2))

    y = np.concatenate((np.zeros(err_normal_train.shape[0]), np.ones(err_bot_train.shape[0])), axis=None)
    x = np.concatenate([err_normal_train, err_bot_train], axis=None)
    train_recon_errors = pd.DataFrame(columns=['recon-err', 'label'], data=np.array([x, y]).T)
    train_recon_errors.to_csv(config['run-dir'] + '/train-reconstruction-errors.csv')

    wandb_hist_norm, wandb_hist_mal = utils.get_wandb_hist_data(err_normal_train, err_bot_train)
    wandb_hist_norm = np.stack([wandb_hist_norm, np.zeros(wandb_hist_norm.shape[0])], axis=1)
    wandb_hist_mal = np.stack([wandb_hist_mal, np.ones(wandb_hist_mal.shape[0])], axis=1)
    table_name = 'reconstruction-error-train'
    wandb.log(
        {table_name: wandb.Table(
            data=np.concatenate([wandb_hist_norm, wandb_hist_mal]).tolist(),
            columns=['reconstruction-error', 'label'])})

    for task in downstream_tasks:
        task_dir = config['run-dir'] + '/downstream-task/' + task
        Path(task_dir).mkdir(parents=True, exist_ok=True)
        downstream_model = downstream_factory.get_downstream_task(task)
        downstream_model.fit(x, y)

        for scenario in test_scenarios:
            utils.test_eval(test_scenarios[scenario], model, downstream_model, scenario, wandb_group_name, run_id, task,
                            config)
