import sys

import numpy as np
import pandas as pd
from wandb.keras import WandbCallback

import utils
from data import loader_factory
from downstream import downstream_factory
from evaluation.evaluation_factory import get_evaluator

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = utils.read_cfg(config_file)
    config = utils.setup_run(config)
    logger = utils.get_logger(**config)

    data_set = config['data']['data-set']
    downstream_tasks = config['downstream']['tasks']
    latent_dim = config['autoencoder']['latent-dim']
    batch_size = config['autoencoder']['batch-size']
    epochs = config['autoencoder']['epochs']
    callbacks = [WandbCallback()] if config['logging']['use-wandb'] else []

    train_norm, train_bot, test_scenarios = loader_factory.get_loader(data_set)(config)

    feature_dim = train_norm.shape
    model = utils.create_model(config, feature_dim=feature_dim)
    model.fit(train_norm, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    err_normal_train, mu_normal_train, log_sig_normal_train = utils.pred(model, train_norm, axis=(1, 2))
    err_bot_train, mu_bot_train, log_sig_bot_train = utils.pred(model, train_bot, axis=(1, 2))

    y = np.concatenate((np.zeros(err_normal_train.shape[0]), np.ones(err_bot_train.shape[0])), axis=None)
    x = np.concatenate([err_normal_train, err_bot_train], axis=None)
    mu = np.concatenate([mu_normal_train, mu_bot_train], axis=0)
    sig = np.concatenate([log_sig_normal_train, log_sig_bot_train], axis=0)
    cols = ['recon-err']
    cols = cols + ['mu' + str(i) for i in range(0, mu.shape[1])]
    cols = cols + ['log-sig' + str(i) for i in range(0, sig.shape[1])] + ['label']
    data = pd.DataFrame(columns=cols, data=np.concatenate([x.reshape(-1, 1), mu, sig, y.reshape(-1, 1)], axis=1))
    item_name = 'reconstruction'
    logger.info('', extra={'data': {'context': 'train', 'entity': 'vae_rnn', 'item': item_name, 'data': data}})

    for task in downstream_tasks:
        downstream_model = downstream_factory.get_downstream_task(task, **config)
        downstream_model.fit(x, y)
        evaluator = get_evaluator(data_set)
        for scenario in test_scenarios:
            evaluator(test_scenarios[scenario], model, downstream_model, scenario, task, **config)
    utils.save_cfg(config)