import os
import time
from pathlib import Path
import datetime
import yaml
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import norm
import scipy.stats as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
import math
import wandb
from data import preprocessing as preproces
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from model.rvae import RVAE


def recon_distribution(y):
    param = norm.fit(y)
    return param


files = [
    'capture20110810.binetflow',
    'capture20110811.binetflow',
    'capture20110812.binetflow',
    'capture20110815.binetflow',
    'capture20110815-2.binetflow',
    'capture20110816.binetflow',
    'capture20110816-2.binetflow',
    'capture20110816-3.binetflow',
    'capture20110817.binetflow',
    'capture20110818.binetflow',
    'capture20110818-2.binetflow',
    'capture20110819.binetflow',
    'capture20110815-3.binetflow'
]

ctu_nr_to_name = {
    1: 'neris',
    2: 'neris',
    3: 'rbot',
    4: 'rbot',
    5: 'virut',
    6: 'menti',
    7: 'sogou',
    8: 'murlo',
    9: 'neris',
    10: 'rbot',
    11: 'rbot',
    12: 'nsis.ay',
    13: 'virut',
}

features = ['Dur', 'Proto', 'SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State', 'sTos', 'dTos', 'TotPkts',
            'TotBytes',
            'SrcBytes']


def get_feature_shape(without_ip):
    return len(get_feature_names(without_ip))


def get_feature_names(without_ip):
    if without_ip:
        return ['Dur', 'Proto', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State', 'sTos', 'dTos', 'TotPkts',
                'TotBytes',
                'SrcBytes']
    else:
        return ['Dur', 'Proto', 'SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State', 'sTos', 'dTos', 'TotPkts',
                'TotBytes',
                'SrcBytes']


def bin_size_for_time_interval(data, seconds=30):
    first = data.iloc[0]
    last = data.iloc[-1]

    duration = last['date'] - first['date']
    return int(duration.seconds / seconds)


def n_second_slices(d, n):
    slices = []
    current_start_time = d.iloc[0]['date']
    current_slice = pd.DataFrame(columns=d.columns)

    start_idx = 0
    for idx, row in d.iterrows():
        if (row['date'] - d.iloc[start_idx]['date']).seconds > n:
            slices.append(d.iloc[start_idx:idx])
            start_idx = idx

    return slices


def fetch_set(ctu_set):
    return pd.read_csv('~/Documents/lbnl/crd/ctu-13/CTU-13-Dataset/' + str(ctu_set) + '/' + files[ctu_set - 1])


def save(frame, name):
    frame.to_csv(name)


def load(ctu_sets, cache_name, ignore_cached=False):
    file = '/Documents/lbnl/crd/ctu-13/CTU-13-Dataset/' + cache_name + '.csv'
    if os.path.isfile(os.path.expanduser(
            "~") + file) and not ignore_cached:
        return pd.read_csv('~' + file, index_col=0)
    else:
        data = fetch_set(ctu_sets[0])
        for ctu_set in ctu_sets[1:]:
            data = pd.concat([data, fetch_set(ctu_set)])
        data = preproces.preprocess_stat_2_CTU(data, 60, False)
        # data = preproces.preprocess_stat_CTU(data)
        # data = preproces.preprocess_CTU(data)
        save(data, cache_name)
        return data


def load_all(ignore_cached=False):
    return load(range(1, 14), 'all', ignore_cached)


def patch_set(ctu_data):
    ctu_data['bot'] = ctu_data['Label'].apply(lambda label: 'bot' in label.lower())
    ctu_data['date'] = ctu_data['StartTime'].apply(lambda ts: datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f'))
    return ctu_data


def feature_selection(ctu_data):
    # return ctu_data[
    #     ['Dur', 'Proto', 'SrcAddr', 'Sport', 'Dir', 'DstAddr', 'Dport', 'State', 'sTos', 'dTos', 'TotPkts', 'TotBytes',
    #      'SrcBytes', 'date']]

    return ctu_data[features]


def split_mixed(data, bin_size, shift=0):
    pass


def encode_labels(ctu_data):
    for cat_var in ['Proto', 'SrcAddr', 'Dir', 'DstAddr', 'State']:
        le = LabelEncoder()
        ctu_data[cat_var] = le.fit_transform(ctu_data[cat_var])
    return ctu_data


def clean_normalize(ctu_data):
    ctu_data[features] = ctu_data[features].apply(pd.to_numeric, errors='coerce')
    ctu_data = ctu_data[~np.isnan(ctu_data[features]).any(axis=1)]
    normed = ctu_data
    normed[features] = normalize(ctu_data[features])
    return normed


def split_mal_norm(ctu_data):
    bot_activity = ctu_data[ctu_data['class'] == 'botnet']
    normal_activity = ctu_data[(ctu_data['class'] == 'normal') | (ctu_data['class'] == 'background')]
    # bot_activity.sort_values('date', ascending=True)
    # normal_activity.sort_values('date', ascending=True)
    return normal_activity, bot_activity


def remove_ylabel(frame):
    return frame.drop(columns=['class'])


def reshape_for_rnn(frame, number_of_slice=5):
    count_x = math.floor(frame.shape[0] / number_of_slice)
    return frame[0:number_of_slice * count_x].reshape(count_x, frame.shape[1], number_of_slice)


def blocked_mal_normal(X, bin_size):
    normal_flows = []
    malicious_flows = []

    for idx in range(0, int(X.shape[0] / bin_size)):
        block = X.iloc[idx * bin_size: idx * bin_size + bin_size]
        is_malicious = True if block[block['bot'] == True].shape[0] != 0 else False

        if is_malicious:
            malicious_flows.append(block[features].values)
        else:
            normal_flows.append(block[features].values)
    normal_flows = np.array(normal_flows)
    normal_flows = normal_flows.reshape(normal_flows.shape[0], normal_flows.shape[1], normal_flows.shape[2], 1)
    malicious_flows = np.array(malicious_flows)
    malicious_flows = malicious_flows.reshape(malicious_flows.shape[0], malicious_flows.shape[1],
                                              malicious_flows.shape[2], 1)
    return normal_flows, malicious_flows


# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * (math.log(p[i] / q[i]) if q[i] > 0 and p[i] > 0 else 0) for i in range(len(p)))


def blocked_evaluation(vae, X, bin_size):
    normal_flows, malicious_flows = blocked_mal_normal(X, bin_size)
    err_norm, mu_norm, log_sig = pred(vae, normal_flows)
    err_mal, mu_mal, log_sig = pred(vae, malicious_flows)

    plt.scatter(mu_norm[:, 0], mu_norm[:, 1], label='encoded - normal flows')
    plt.scatter(mu_mal[:, 0], mu_mal[:, 1], color=['orange' for i in range(0, mu_mal.shape[0])],
                label='encoded - malicious flows')
    plt.show()

    plt.hist(err_norm, bins=100, label='normal')
    plt.hist(err_mal, bins=100, label='contains anomaly')
    plt.legend()
    plt.show()

    kde_n = KDEUnivariate(err_norm)
    kde_n.fit()

    kde_an = KDEUnivariate(err_mal)
    kde_an.fit()

    xs = np.linspace(0, max(err_norm.max(), err_mal.max()), 750)
    ys_n = kde_n.evaluate(xs)
    ys_an = kde_an.evaluate(xs)

    plt.plot(xs, ys_n, label='normal')
    plt.plot(xs, ys_an, label='contains anomaly')
    plt.legend()
    plt.show()


def pred(vae, X, axis=1):
    mu, log_sig, z = vae.encoder.predict(X)
    rec = vae.decoder.predict(z)
    # err = ((X - rec) ** 2).reshape(X.shape[0], X.shape[1] * X.shape[2])
    err = ((X - rec) ** 2).sum(axis=axis)
    # err = np.diag(err @ err.T)
    return err, mu, log_sig


def reverse(data):
    return np.flip(data, 1)


def add_noise(data):
    noise = np.random.normal(0, .1, data.shape)
    return data + noise


def time_shift(data):
    shp = data.shape
    data = data.reshape(shp[0] * shp[-1], shp[1])
    offset = math.ceil(5 / 2)
    shifted = data[offset:, ]
    count = math.floor(shifted.shape[0] / 5)
    shifted = shifted[0:5 * count]
    return shifted.reshape(count, shp[1], 5)


def generate_from_rvae(data):
    feature_dim = data.shape
    latent_dim = 5

    encoder_inputs = keras.Input(shape=(feature_dim[1], feature_dim[2]))
    encoder_layers = layers.GRU(units=10)(encoder_inputs)
    encoder_layers = layers.LeakyReLU(alpha=0.01)(encoder_layers)

    latent_inputs = keras.Input(shape=(latent_dim))
    decoder_layers = layers.Dense(feature_dim[2] * feature_dim[1])(latent_inputs)
    decoder_layers = layers.Reshape((feature_dim[1], feature_dim[2]))(decoder_layers)
    decoder_layers = layers.GRU(units=10, activation='relu', return_sequences=True)(decoder_layers)
    decoder_layers = layers.Dense(5)(decoder_layers)

    vae = RVAE(encoder_inputs, encoder_layers, latent_inputs, decoder_layers, kl_weight=0.75)
    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(data, epochs=500, batch_size=128)
    samples = np.random.normal(0, 1, (data.shape[0], 1, latent_dim))
    augmented = np.array([vae.decoder.predict(sample) for sample in samples]).reshape(data.shape)
    augmented = np.maximum(augmented, np.zeros(shape=augmented.shape))
    return augmented


def augment_reverse(data):
    return np.concatenate([data, reverse(data)])


def augment_noise(data):
    return np.concatenate([data, add_noise(data)])


def augment_time_shift(data):
    return np.concatenate([data, time_shift(data)])


def augment_rvae_generate(data):
    return np.concatenate([data, generate_from_rvae(data)])


def augment(data, type):
    if type == 'reverse':
        return augment_reverse(data)
    if type == 'noise':
        return augment_noise(data)
    if type == 'noise-replace':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_noise(data)
    if type == 'reverse-replace':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_reverse(data)
    if type == 'time-shift':
        return augment_time_shift(data)
    if type == 'rvae-generate':
        return augment_rvae_generate(data)
    if type == 'rvae-generate-half':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_rvae_generate(data)
    if type == 'half-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
    return data


def predict(vae, downstream_model, X, axis=1):
    err, mu, log_sig = pred(vae, X, axis=axis)
    return np.array([downstream_model.predict(er) for er in err]), err


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    DISTRIBUTIONS2 = [st.invgamma, st.nct]

    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk,
        # st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
        st.invgauss,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            params = distribution.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
            # if axis pass in add to plot
            try:
                if ax:
                    pd.Series(pdf, x).plot(ax=ax)
            except Exception:
                pass

            # identify if this distribution is better
            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse

        except Exception:
            print('exception')
            pass

    return (best_distribution.name, best_params)


def plot_train_hist(recon_err_norm, recon_err_mal, title, log_wandb=True):
    plt.hist(recon_err_norm, bins=200, color='green')
    plt.hist(recon_err_mal, bins=200, color='red')
    plt.title('title')
    plt.xlabel('Error (MSE)')
    plt.ylabel('Count')
    plt.show()

    data_norm = np.stack([recon_err_norm, np.zeros(recon_err_norm.shape)], axis=1)
    data_mal = np.stack([recon_err_mal, np.zeros(recon_err_mal.shape)], axis=1)

    wandb.log(
        {'reconstruction-error-test': wandb.Table(data=np.concatenate([data_norm, data_mal]).tolist(),
                                                  columns=['reconstruction-error', 'label'])})

    pass


def get_wandb_hist_data(recon_err_norm, recon_err_mal):
    data_norm = np.stack([recon_err_norm, np.zeros(recon_err_norm.shape)], axis=1)
    data_mal = np.stack([recon_err_mal, np.zeros(recon_err_mal.shape)], axis=1)

    if data_norm.shape[0] > 5000:
        data_norm = data_norm[np.random.randint(data_norm.shape[0], size=5000), :]

    size = 1000 - math.min(data_norm.shape[0], 5000)
    if data_mal.shape[0] > 5000:
        hist_mal = data_mal[np.random.randint(data_mal.shape[0], size=size), :]


def simple_eval(test_data, vae, downstream_model):
    test_norm, test_bot = split_mal_norm(test_data)
    test_norm = remove_ylabel(test_norm)
    test_norm = reshape_for_rnn(test_norm.values)
    test_bot = remove_ylabel(test_bot)
    test_bot = reshape_for_rnn(test_bot.values)

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = predict(vae, downstream_model, np.concatenate([test_norm, test_bot]), axis=(1, 2))
    preds = preds.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds)
    prec = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    print('#normal: ' + str(test_norm.shape[0]) + ' / #bot: ' + str(test_bot.shape[0]))
    print('prec:' + str(prec))
    print('rec:' + str(recall))
    print('f1: ' + str(f1))
    print(str(conf_matrix))


def test_eval(test_data, vae, downstream_model, scenario, file_prefix, run_id, task, config):
    print('test eval for scenario ' + str(scenario))
    test_norm, test_bot = split_mal_norm(test_data)
    test_norm = remove_ylabel(test_norm)
    test_norm = reshape_for_rnn(test_norm.values)
    test_bot = remove_ylabel(test_bot)
    test_bot = reshape_for_rnn(test_bot.values)

    y = np.concatenate((np.zeros(test_norm.shape[0]), np.ones(test_bot.shape[0])), axis=None)
    preds, recon_err = predict(vae, downstream_model, np.concatenate([test_norm, test_bot]), axis=(1, 2))
    preds = preds.astype(int).reshape(y.shape)
    conf_matrix = confusion_matrix(y, preds)
    prec = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    print('#normal: ' + str(test_norm.shape[0]) + ' / #bot: ' + str(test_bot.shape[0]))
    print('prec:' + str(prec))
    print('rec:' + str(recall))
    print('f1: ' + str(f1))
    print(str(conf_matrix))

    wandb.run.summary["Precision_task_" + task + "_scenario_" + str(scenario)] = prec
    wandb.run.summary["Recall_task_" + task + "_scenario_" + str(scenario)] = recall
    wandb.run.summary["F1_task_" + task + "_scenario_" + str(scenario)] = f1
    wandb.sklearn.plot_confusion_matrix(y, preds, ['normal', 'botnet'])
    hist_data = np.stack([recon_err, y], axis=1)

    hist_norm = hist_data[0:test_norm.shape[0]]
    if hist_norm.shape[0] > 5000:
        hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000), :]

    hist_mal = hist_data[test_norm.shape[0]:, ]
    if hist_mal.shape[0] > 5000:
        hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000), :]

    table_name = 'reconstruction-error-test-taks-' + task + '_scenario-' + str(scenario)
    wandb.log(
        {table_name: wandb.Table(
            data=np.concatenate([hist_norm, hist_mal]).tolist(),
            columns=['reconstruction-error', 'label'])})
    log = pd.DataFrame(columns=['recon-err', 'prediction', 'label'], data=np.stack([recon_err, preds, y], axis=1))
    log_name = file_prefix + '-reconstruction-error-' + str(scenario)
    log.to_csv(config['run-dir'] + '/downstream-task/' + task + '/' + log_name + '.csv')


def read_cfg(cfg_file):
    cfg = None
    with open(cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    if cfg is None:
        print('no cfg present')
    return cfg


def setup_run(cfg):
    d = datetime.today()
    cfg['run-id'] = cfg['experiment-name'] + '-' + d.strftime("%Y-%m-%d-%H-%M-%S")
    cfg['run-dir'] = "./runs/" + cfg['experiment-name'] + '/' + cfg['run-id']
    Path(cfg['run-dir']).mkdir(parents=True, exist_ok=True)
    return cfg


def create_autoencoder(cfg, feature_dim=None, **kwargs):
    sub_type = cfg['sub-type']
    latent_dim = cfg['config']['latent-dimension']
    enc_layers = cfg['config']['encoding-layers']
    dec_layers = cfg['config']['decoding-layers']
    optimizer = cfg['config']['optimizer']['class-name']
    # optimizer_args = cfg['config']['optimizer']['args']
    optimizer_args = {}

    encoder_inputs = keras.Input(shape=(feature_dim[1], feature_dim[2]))
    encoder_layers = getattr(layers, enc_layers[0]['class-name'])(**enc_layers[0]['arguments'])(encoder_inputs)
    for layer in enc_layers[1:]:
        encoder_layers = getattr(layers, layer['class-name'])(**layer['arguments'])(encoder_layers)

    latent_inputs = keras.Input(shape=(latent_dim))
    decoder_layers = layers.Dense(feature_dim[2] * feature_dim[1])(latent_inputs)
    decoder_layers = layers.Reshape((feature_dim[1], feature_dim[2]))(decoder_layers)
    for layer in dec_layers:
        decoder_layers = getattr(layers, layer['class-name'])(**layer['arguments'])(decoder_layers)
    decoder_layers = layers.Dense(feature_dim[2])(decoder_layers)

    vae = RVAE(encoder_inputs, encoder_layers, latent_inputs, decoder_layers)
    vae.compile(optimizer=getattr(keras.optimizers, optimizer)(**optimizer_args))
    return vae


def create_model(cfg, **kwargs):
    model_type = cfg['model']['type']
    if model_type == 'autoencoder':
        return create_autoencoder(cfg['model'], **kwargs)
    return None


if __name__ == '__main__':
    config = read_cfg('./config/template.yaml')
    model = create_model(config, **{'feature_dim': (None, 23, 5)})
    print(model.summary())

    # train_sets = [3, 4, 5, 7, 10, 11, 12, 13]
    # test_sets = [1, 2, 6, 8, 9]
    # train = load(train_sets, 'train-def-medium', True)
    # test = data[~msk]
    # test = load(test_sets, 'test-def-medium', True)
    # test_scenarios = {scenario: load([scenario], 'scenario-test-medium-' + str(scenario), True) for scenario in
    #                   test_sets}
