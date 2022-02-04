import math
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback

import utils
import wandb
from model.rvae import RVAE

if __name__ == '__main__':
    train_sets = [3, 4, 5, 7, 10, 11, 12, 13]
    test_sets = [1, 2, 6, 8, 9]
    wandb_group_name = sys.argv[1] if len(sys.argv) >= 2 else 'test'
    augment_type = sys.argv[2] if len(sys.argv) >= 3 else None
    wandb.init(project="ctu-13-augment-test", entity="lbl-crd",
               group=str(wandb_group_name))
    run_id = time_s = str(round(time.time()))

    #
    latent_dim = 5
    # data = utils.encode_labels(utils.patch_set(utils.fetch_set(ctu_data_set)))
    # data = utils.load_all()
    # data = utils.load_all(True)
    # data = preproces.preprocess_stat_2_CTU(utils.fetch_set(ctu_data_set), 60, False)
    # msk = np.random.rand(len(data)) < 0.8
    # train = data[msk]
    train = utils.load(train_sets, 'train-def-medium', False)
    # test = data[~msk]
    test = utils.load(test_sets, 'test-def-medium', False)
    test_scenarios = {scenario: utils.load([scenario], 'scenario-test-medium-' + str(scenario), False) for scenario in
                      test_sets}
    d = [test_scenarios[d].columns.values for d in test_scenarios.keys()]
    common_attributes = set.intersection(*map(set, d))
    train = train.drop(columns=[feat for feat in train.columns if feat not in common_attributes])
    test = test.drop(columns=[feat for feat in test.columns if feat not in common_attributes])
    test_scenarios = {
        scenario: test_scenarios[scenario].drop(
            columns=[feat for feat in test_scenarios[scenario].columns if feat not in common_attributes])
        for scenario in test_scenarios.keys()}

    train_norm, train_bot = utils.split_mal_norm(train)
    count_x = math.floor(train_norm.shape[0] / 5)
    train_norm = utils.remove_ylabel(train_norm)
    train_norm = utils.reshape_for_rnn(train_norm.values)
    train_norm = utils.augment(train_norm, augment_type)

    train_bot = utils.remove_ylabel(train_bot)
    train_bot = utils.reshape_for_rnn(train_bot.values)

    test_norm, test_bot = utils.split_mal_norm(test)
    test_norm = utils.remove_ylabel(test_norm)
    test_norm = utils.reshape_for_rnn(test_norm.values)
    test_bot = utils.remove_ylabel(test_bot)
    test_bot = utils.reshape_for_rnn(test_bot.values)

    feature_dim = train_norm.shape

    encoder_inputs = keras.Input(shape=(feature_dim[1], feature_dim[2]))
    encoder_layers = layers.GRU(units=10)(encoder_inputs)
    encoder_layers = layers.LeakyReLU(alpha=0.01)(encoder_layers)

    latent_inputs = keras.Input(shape=(latent_dim))
    decoder_layers = layers.Dense(feature_dim[2] * feature_dim[1])(latent_inputs)
    decoder_layers = layers.Reshape((feature_dim[1], feature_dim[2]))(decoder_layers)
    decoder_layers = layers.GRU(units=10, activation='relu', return_sequences=True)(decoder_layers)
    decoder_layers = layers.Dense(5)(decoder_layers)

    vae = RVAE(encoder_inputs, encoder_layers, latent_inputs, decoder_layers)
    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(train_norm, epochs=500, batch_size=128, callbacks=[WandbCallback()])

    err_normal_train, mu_normal_train, log_sig_normal_train = utils.pred(vae, train_norm, axis=(1, 2))
    err_bot_train, mu_bot_train, log_sig_bot_train = utils.pred(vae, train_bot, axis=(1, 2))

    err_normal, mu_normal, log_sig_normal = utils.pred(vae, test_norm, axis=(1, 2))
    err_bot, mu_bot, log_sig_bot = utils.pred(vae, test_bot, axis=(1, 2))

    normal_dist_n, abparams_norm = utils.best_fit_distribution(err_normal_train)
    normal_dist = getattr(st, normal_dist_n)

    malicious_dist_n, abparams_mal = utils.best_fit_distribution(err_bot_train)
    malicious_dist = getattr(st, malicious_dist_n)

    max_x = max(err_normal.max(), err_bot_train.max())
    max_x = int(max_x * 1.5)
    xs = np.linspace(0, max_x, 1000)
    ys_normal = normal_dist.pdf(xs, loc=abparams_norm[-2], scale=abparams_norm[-1], *abparams_norm[:-2])
    ys_malicious = malicious_dist.pdf(xs, loc=abparams_mal[-2], scale=abparams_mal[-1], *abparams_mal[:-2])

    plt.scatter(mu_normal_train[:, 0], mu_normal_train[:, 1], label='encoded - normal (train)')
    plt.scatter(mu_bot_train[:, 0], mu_bot_train[:, 1], color=['orange' for i in range(0, mu_bot_train.shape[0])],
                label='encoded - malicious (train)')
    plt.legend()
    plt.show()

    wandb_normal = mu_normal_train[np.random.randint(mu_normal_train.shape[0], size=min(500, mu_normal_train.shape[0])),
                   :]
    wandb_malicious = mu_bot_train[np.random.randint(mu_bot_train.shape[0], size=min(500, mu_bot_train.shape[0])), :]
    wandb_data = np.append(wandb_normal, [['normal - train'] for i in range(0, wandb_normal.shape[0])], axis=1)
    wandb_data = np.concatenate(
        (wandb_data, np.append(wandb_malicious, [['malicious - train'] for i in range(0, wandb_malicious.shape[0])],
                               axis=1)), axis=0)
    wandb_cols = ['mu_' + str(i) for i in range(0, wandb_normal.shape[1])] + ['label']

    wandb.log(
        {'latent-mus': wandb.Table(data=wandb_data, columns=wandb_cols)})

    plt.plot(xs, ys_normal, label='normal flows (train)')
    plt.plot(xs, ys_malicious, label='malicious flows')
    plt.legend()
    plt.title('Reconstruction error distribution')
    plt.show()

    normal_sample = err_normal_train[
        np.random.randint(err_normal_train.shape[0], size=min(1000, err_normal_train.shape[0]))]
    mal_sample = err_bot_train[np.random.randint(err_bot_train.shape[0], size=min(1000, err_bot_train.shape[0]))]
    data_n = [[er, 'normal'] for er in normal_sample]
    data_an = [[er, 'malicious'] for er in mal_sample]
    wandb.log(
        {'reconstruction-error': wandb.Table(data=(data_n + data_an), columns=['reconstruction-error', 'label'])})

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = utils.predict(vae, normal_dist, malicious_dist, np.concatenate([test_norm, test_bot]),
                                     abparams_norm=abparams_norm, abparams_mal=abparams_mal, axis=(1, 2))
    preds = preds.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds)
    prec = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    print('#normal: ' + str(test_norm.shape[0]) + ' / #bot: ' + str(test_bot.shape[0]))
    print('prec:' + str(prec))
    print('rec:' + str(recall))
    print('f1: ' + str(f1))
    print('best norm dist: ' + normal_dist_n)
    print('best mal dist: ' + malicious_dist_n)
    print(str(conf_matrix))

    wandb.run.summary["Precision"] = prec
    wandb.run.summary["Recall"] = recall
    wandb.run.summary["F1"] = f1
    wandb.sklearn.plot_confusion_matrix(y, preds, ['normal', 'botnet'])
    hist_data = np.stack([recon_err, y], axis=1)

    hist_norm = hist_data[0:test_norm.shape[0]]
    if hist_norm.shape[0] > 5000:
        hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000), :]

    hist_mal = hist_data[test_norm.shape[0]:, ]
    if hist_mal.shape[0] > 5000:
        hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000), :]

    wandb.log(
        {'reconstruction-error-test': wandb.Table(data=np.concatenate([hist_norm, hist_mal]).tolist(),
                                                  columns=['reconstruction-error', 'label'])})
    for scenario in test_scenarios:
        utils.test_eval(test_scenarios[scenario], vae, normal_dist, malicious_dist, abparams_norm, abparams_mal,
                        scenario,
                        wandb_group_name, run_id)
