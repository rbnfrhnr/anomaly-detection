import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from model.vae import VAE
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import utils
import sys
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import scipy.stats as st

if __name__ == '__main__':
    train_sets = [3, 4, 5, 7, 10, 11, 12, 13]
    test_sets = [1, 2, 6, 8, 9]
    ctu_data_set = int(sys.argv[1]) if 1 < len(sys.argv) else 3
    wandb.init(project="ctu-13-augment-test", entity="lbl-crd",
               group="vae-dnn-test" + str(ctu_data_set) + "-" + str(utils.ctu_nr_to_name[ctu_data_set]))
    #
    latent_dim = 10
    # data = utils.encode_labels(utils.patch_set(utils.fetch_set(ctu_data_set)))
    # data = utils.load_all()
    # data = utils.load_all(True)
    # data = preproces.preprocess_stat_2_CTU(utils.fetch_set(ctu_data_set), 60, False)
    # msk = np.random.rand(len(data)) < 0.8
    # train = data[msk]
    train = utils.load(train_sets, 'train-def', False)
    # test = data[~msk]
    test = utils.load(test_sets, 'test-def', False)
    train = train.drop(columns=[feat for feat in train.columns if feat not in test.columns])
    test = test.drop(columns=[feat for feat in test.columns if feat not in train.columns])

    train_norm, train_bot = utils.split_mal_norm(train)
    train_norm = utils.remove_ylabel(train_norm)
    train_bot = utils.remove_ylabel(train_bot)

    test_norm, test_bot = utils.split_mal_norm(test)
    test_norm = utils.remove_ylabel(test_norm)
    test_bot = utils.remove_ylabel(test_bot)

    feature_dim = train_norm.shape[1]
    # data.sort_values('date', ascending=True)

    encoder_inputs = keras.Input(shape=feature_dim)
    # encoder_layers = layers.Reshape((bin_size * feature_dim, 1))(encoder_inputs)
    encoder_layers = layers.Dense(units=94)(encoder_inputs)
    # encoder_layers = layers.Dense(units=30, activation='relu')(encoder_layers)
    # encoder_layers = layers.Dense(units=1024)(encoder_layers)
    encoder_layers = layers.Dense(units=30)(encoder_layers)
    encoder_layers = layers.LeakyReLU(alpha=0.01)(encoder_layers)

    latent_inputs = keras.Input(shape=latent_dim)
    decoder_layers = layers.Dense(units=30)(latent_inputs)
    # decoder_layers = layers.Dense(units=94, activation='relu')(decoder_layers)
    decoder_layers = layers.Dense(units=94)(decoder_layers)
    decoder_layers = layers.LeakyReLU(alpha=0.01)(decoder_layers)
    decoder_layers = layers.Dense(feature_dim)(decoder_layers)

    vae = VAE(encoder_inputs, encoder_layers, latent_inputs, decoder_layers)
    vae.compile(optimizer=keras.optimizers.Adam())

    # X_train = data_norm[utils.get_feature_names(without_src_ip)].values
    # X_train = X_train[0:bin_size * int(X_train.shape[0] / bin_size)]
    # X_train = X_train.reshape(int(X_train.shape[0] / bin_size), bin_size, feature_dim, 1)
    vae.fit(train_norm, epochs=50, batch_size=128, callbacks=[WandbCallback()])
    # vae.fit(bla, epochs=15, batch_size=128)
    # utils.blocked_evaluation(vae, data_sub, bin_size)

    err_normal_train, mu_normal_train, log_sig_normal_train = utils.pred(vae, train_norm)
    err_bot_train, mu_bot_train, log_sig_bot_train = utils.pred(vae, train_bot)

    err_normal, mu_normal, log_sig_normal = utils.pred(vae, test_norm)
    err_bot, mu_bot, log_sig_bot = utils.pred(vae, test_bot)

    #
    normal_dist_n, abparams_norm = utils.best_fit_distribution(err_normal_train)
    normal_dist = getattr(st, normal_dist_n)

    # normal_dist = KDEUnivariate(err_normal_train)
    # normal_dist.fit()

    malicious_dist_n, abparams_mal = utils.best_fit_distribution(err_bot_train)
    malicious_dist = getattr(st, malicious_dist_n)

    # malicious_dist = KDEUnivariate(err_bot_train)
    # malicious_dist.fit()

    max_x = max(err_normal.max(), err_bot_train.max())
    max_x = int(max_x * 1.5)
    xs = np.linspace(0, max_x, 1000)
    # ys_normal = normal_dist.evaluate(xs)
    ys_normal = normal_dist.pdf(xs, loc=abparams_norm[-2], scale=abparams_norm[-1], *abparams_norm[:-2])
    # ys_malicious = malicious_dist.evaluate(xs)
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

    # wandb.log({'kde-normal-anomaly': wandb.plot.line_series(
    #     xs=xs,
    #     ys=[ys_normal, ys_malicious],
    #     keys=['normal flows', 'malicious flows'],
    #     title='KDE estimated distribution of normal/anomaly (train)',
    #     xname='reconstruction error squared'
    # )})

    data_n = [[er, 'normal'] for er in np.random.choice(err_normal_train, size=min(1000, err_normal_train.shape[0]))]
    data_an = [[er, 'malicious'] for er in np.random.choice(err_bot_train, size=min(1000, err_normal_train.shape[0]))]
    wandb.log(
        {'reconstruction-error': wandb.Table(data=(data_n + data_an), columns=['reconstruction-error', 'label'])})

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = utils.predict(vae, normal_dist, malicious_dist, pd.concat([test_norm, test_bot]),
                                     abparams_norm=abparams_norm, abparams_mal=abparams_mal)
    preds = preds.astype(int).reshape(y.shape)
    # correct = np.array([(preds[i] == y[i]) for i in range(0, len(y))]).astype(int)
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
