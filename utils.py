import datetime
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
import yaml
from keras import backend
from tensorflow import keras
from tensorflow.keras import layers

from logger.data_logger import DataHandler
from logger.wandb_logger import WandbHandler
from logger.stream_handler import CustomStreamHandler
from model.rvae import RVAE

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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def save(frame, name):
    frame.to_csv(name)


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
    return frame[0:number_of_slice * count_x].reshape(count_x, number_of_slice, frame.shape[1])


# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * (math.log(p[i] / q[i]) if q[i] > 0 and p[i] > 0 else 0) for i in range(len(p)))


def pred(vae, X, axis=(1, 2)):
    if X.shape[0] > 0:
        mu, log_sig, z = vae.encoder.predict(X)
        rec = vae.decoder.predict(z)
        err = ((X - rec) ** 2).sum(axis=axis)
        return err, mu, log_sig
    return np.array([]), np.array([]), np.array([])


def reverse(data):
    return np.flip(data, 1)


def add_noise(data):
    noise = np.random.normal(0, .025, data.shape)
    return data + noise


def time_shift(data):
    shp = data.shape
    data = data.reshape(shp[0] * shp[1], shp[-1])
    offset = math.ceil(shp[1] / 2)
    shifted = data[offset:, ]
    count = math.floor(shifted.shape[0] / shp[1])
    shifted = shifted[0:shp[1] * count]
    return shifted.reshape(count, shp[1], shp[-1])


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
    decoder_layers = layers.Dense(23)(decoder_layers)

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


def augment2(data, type):
    if type == 'reverse':
        return reverse(data)
    if type == 'noise':
        return add_noise(data)
    if type == 'noise-replace':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_noise(data)
    if type == 'reverse-replace':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_reverse(data)
    if type == 'reverse-fifth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
        return augment_reverse(data)
    if type == 'reverse-tenth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
        return augment_reverse(data)
    if type == 'time-shift':
        return augment_time_shift(data)
    if type == 'time-shift-half':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_time_shift(data)
    if type == 'time-shift-fifth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
        return augment_time_shift(data)
    if type == 'time-shift-tenth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
        return augment_time_shift(data)
    if type == 'rvae-generate':
        return augment_rvae_generate(data)
    if type == 'rvae-generate-half':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_rvae_generate(data)
    if type == 'half-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
    if type == 'fifth-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
    if type == 'tenth-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
    return data


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
    if type == 'reverse-fifth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
        return augment_reverse(data)
    if type == 'reverse-tenth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
        return augment_reverse(data)
    if type == 'time-shift':
        return augment_time_shift(data)
    if type == 'time-shift-half':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_time_shift(data)
    if type == 'time-shift-fifth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
        return augment_time_shift(data)
    if type == 'time-shift-tenth':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
        return augment_time_shift(data)
    if type == 'rvae-generate':
        return augment_rvae_generate(data)
    if type == 'rvae-generate-half':
        data = data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
        return augment_rvae_generate(data)
    if type == 'half-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 2)), :]
    if type == 'fifth-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 5)), :]
    if type == 'tenth-data':
        return data[np.random.randint(data.shape[0], size=math.floor(data.shape[0] / 10)), :]
    return data


def predict(vae, downstream_model, X, axis=(1, 2)):
    err, mu, log_sig = pred(vae, X, axis=axis)
    return downstream_model.predict(err), err


def predict2(vae, downstream_model, X, axis=(1, 2)):
    err, mu, log_sig = pred(vae, X, axis=axis)
    score_norm, score_mal = downstream_model.predict2(err)
    return score_norm, score_mal, err


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


def get_wandb_hist_data(hist_norm, hist_mal):
    if hist_norm.shape[0] > 5000:
        # hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000), :]
        hist_norm = hist_norm[np.random.randint(hist_norm.shape[0], size=5000)]

    if hist_mal.shape[0] > 5000:
        # hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000), :]
        hist_mal = hist_mal[np.random.randint(hist_mal.shape[0], size=5000)]
    return hist_norm, hist_mal


def set_seeds(cfg):
    # Seed value
    default_seed = int(round(999999 * random.random()) + 1)
    seed_value = cfg['global-seed'] if 'global-seed' in cfg else None
    seed_value = seed_value if seed_value is not None else default_seed
    cfg['global-seed'] = seed_value

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # for later versions:
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def read_cfg(cfg_file):
    with open(cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    if cfg is None:
        print('no cfg present')
    return cfg


def save_cfg(cfg):
    run_dir = cfg['run-dir']
    with open(r'' + run_dir + '/run-config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)


def setup_run(cfg):
    d = datetime.today()
    log_location = cfg['logging']['log-location']
    logger_name = cfg['logging']['logger-name']
    use_wandb = cfg['logging']['use-wandb']
    use_gpu = cfg['use-gpu']
    cfg['run-id'] = cfg['experiment-name'] + '-' + d.strftime("%Y-%m-%d-%H-%M-%S")
    cfg['run-dir'] = log_location + "/" + cfg['experiment-name'] + '/' + cfg['run-id']
    Path(cfg['run-dir']).mkdir(parents=True, exist_ok=True)
    set_seeds(cfg)
    save_cfg(cfg)

    lgr = logging.getLogger(logger_name)
    lgr.setLevel(logging.DEBUG)
    lgr.addHandler(DataHandler(cfg['run-dir']))
    lgr.addHandler(CustomStreamHandler(cfg['run-dir']))
    if use_wandb:
        lgr.addHandler(WandbHandler(**cfg))

    print(backend._get_available_gpus())
    if use_gpu:
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        backend.set_session(sess)
    return cfg


def get_logger(**cfg):
    logger_name = cfg['logging']['logger-name']
    return logging.getLogger(logger_name)


def create_autoencoder(cfg, feature_dim=None, latent_dim=5, **kwargs):
    sub_type = cfg['sub-type']
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
        return create_autoencoder(cfg['model'], latent_dim=cfg['autoencoder']['latent-dim'], **kwargs)
    return None


if __name__ == '__main__':
    config = read_cfg('./config/template.yaml')
    config = setup_run(config)
    # model = create_model(config, **{'feature_dim': (None, 23, 5)})
    # print(model.summary())
    # train_sets = [3, 4, 5, 7, 10, 11, 12, 13]
    # test_sets = [1, 2, 6, 8, 9]
    # test = data[~msk]
    # test = load(test_sets, 'test-def-medium', True)
    # test_scenarios = {scenario: load([scenario], 'scenario-test-medium-' + str(scenario), True) for scenario in
    #                   test_sets}
