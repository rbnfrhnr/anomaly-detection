import math
import multiprocessing
import os
from datetime import datetime
from multiprocessing.pool import Pool

import numpy as np
from sklearn.neighbors import KernelDensity
from tensorflow import keras


def plot_density(kde_model, x_norm, x_mal):
    x_min = min(np.min(x_norm), np.min(x_mal))
    x_max = max(np.max(x_norm), np.max(x_mal))
    xx = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    plt.plot(xx, np.exp(kde_model.norm_pdf.score_samples(xx)))
    plt.plot(xx, np.exp(kde_model.mal_pdf.score_samples(xx)))
    plt.hist(x_norm, bins=100, density=True)
    plt.hist(x_mal, bins=100, density=True)
    d = datetime.today()
    suffix = d.strftime("%Y-%m-%d-%H-%M-%S")
    # plt.savefig('./runs/kdes/kde-' + suffix + '.png')
    plt.show()


class T():
    def __init__(self, kde):
        self.kde = kde

    def predi(self, x):
        return self.kde.score_samples(x.reshape(-1, 1))


def set_niceness(nval):
    os.nice(nval)


class KDESklearnDownstream(keras.Model):

    def __init__(self, **params):
        super(KDESklearnDownstream, self).__init__()
        self.norm_pdf = None
        self.mal_pdf = None

    def fit(self, x, y, plot_hist=True, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        self.norm_pdf = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(x_norm.reshape(-1, 1))
        self.mal_pdf = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(x_mal.reshape(-1, 1))

        if plot_hist:
            plot_density(self, x_norm, x_mal)

        return self

    def predict(self, x, parallel=True, **kwargs):
        if parallel:
            norm_score, mal_score = self._predict_parallel(x, **kwargs)
        else:
            norm_score = self.norm_pdf.score_samples(x.reshape(-1, 1))
            mal_score = self.mal_pdf.score_samples(x.reshape(-1, 1))

        return mal_score > norm_score

    def predict2(self, x, parallel=True, **kwargs):
        if parallel:
            norm_score, mal_score = self._predict_parallel(x, **kwargs)
        else:
            norm_score = self.norm_pdf.score_samples(x.reshape(-1, 1))
            mal_score = self.mal_pdf.score_samples(x.reshape(-1, 1))

        return norm_score, mal_score

    def _predict_parallel(self, x, **kwargs):
        t = T(self.norm_pdf)
        t2 = T(self.mal_pdf)

        cores_to_use = math.floor(0.9 * multiprocessing.cpu_count())
        chnk_size = max(round(x.shape[0] / cores_to_use), 200000)

        with Pool(cores_to_use, initializer=set_niceness, initargs=[19]) as p:
            norm_score = p.map(t.predi, x, chunksize=chnk_size)
        norm_score = np.array(norm_score)

        with Pool(cores_to_use, initializer=set_niceness, initargs=[19]) as p:
            mal_score = p.map(t2.predi, x, chunksize=chnk_size)
        mal_score = np.array(mal_score)
        return norm_score, mal_score


if __name__ == '__main__':
    model = KDESklearnDownstream()
    model.compile()
    train = np.concatenate([np.random.random(100000), np.random.random(100000) + 1])
    # y = np.array([0, 0, 1, 1])
    y = np.concatenate([np.zeros(100000), np.ones(100000)])
    model.fit(train, y)
    print(str(model.predict(np.array([1.5, 3.5]))))
    import matplotlib.pyplot as plt

    xx = np.linspace(0, 5, 100)
    plt.plot(xx, model.predict(xx))
    plt.show()
