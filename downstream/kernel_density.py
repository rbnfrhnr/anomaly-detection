import math

from tensorflow import keras
import numpy as np
import tensorflow as tf
import scipy.stats as st
import utils
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import multiprocessing
import os


class TempTest1:

    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return self.a + x


class Temp:
    def __init__(self, a):
        self.a = a

    def test(self):
        return TempTest1(self.a)


def InterfaceFunc(f, x):
    mypool = Pool(4)
    return list(mypool.map(f.a.evaluate, x))


def patch(kde):
    return kde.evaluate


def f(kde, x):
    return kde.evaluate(x)


class T():
    def __init__(self, kde):
        self.kde = kde

    def predi(self, x):
        return self.kde.score_samples(x.reshape(-1, 1))
        # return self.kde.evaluate(x)


def set_niceness(nval):
    os.nice(nval)


class KDEDownstream(keras.Model):

    def __init__(self, **params):
        super(KDEDownstream, self).__init__()
        self.norm_pdf = None
        self.mal_pdf = None

    def fit(self, x, y, **kwargs):
        max_len = 1000000
        x_norm = x[np.where(y == 0)]
        x_norm = x_norm if x_norm.shape[0] < max_len else np.random.choice(x_norm, max_len)

        # bins_norm = min(x_norm.shape[0], 100000)
        # bin_norm = (np.histogram(x_norm, bins_norm, weights=x_norm)[0])

        x_mal = x[np.where(y == 1)]
        x_mal = x_mal if x_mal.shape[0] < max_len else np.random.choice(x_mal, max_len)
        # bins_mal = min(x_mal.shape[0], 100000)
        # bin_mal = (np.histogram(x_mal, bins_mal, weights=x_mal)[0])

        self.norm_pdf = KDEUnivariate(x_norm)
        self.norm_pdf.fit()
        # self.norm_pdf = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(x_norm.reshape(-1, 1))
        # self.norm_pdf = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_norm.reshape(-1, 1))

        self.mal_pdf = KDEUnivariate(x_mal)
        self.mal_pdf.fit()
        # self.mal_pdf = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(x_mal.reshape(-1, 1))

        x_min = min(np.min(x_norm), np.min(x_mal))
        x_max = max(np.max(x_norm), np.max(x_mal))

        # xx = np.linspace(min(x), max(x), 200).reshape(-1, 1)
        xx = np.linspace(x_min, x_max, 200)
        # plt.plot(xx, np.exp(self.norm_pdf.score_samples(xx)))
        # plt.plot(xx, self.norm_pdf.evaluate(xx))
        # plt.plot(xx, np.exp(self.mal_pdf.score_samples(xx)))
        plt.plot(xx, self.mal_pdf.evaluate(xx))
        plt.plot(xx, self.mal_pdf.evaluate(xx))
        plt.hist(x_norm, bins=100, density=True)
        plt.hist(x_mal, bins=100, density=True)
        plt.show()
        # self.mal_pdf = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_mal.reshape(-1, 1))
        return self

    def predict(self, x, **kwargs):
        # norm_score = self.norm_pdf.score_samples(x.reshape(-1, 1))
        # t1 = Temp(self.norm_pdf).test()
        # norm_score = InterfaceFunc(t1.a, x)
        # with Pool(5) as p:
        # norm_score = p.map(patch(self.norm_pdf), x, 3000)
        # norm_score = p.apply(f, args=[self.norm_pdf, x])
        # norm_score = [self.norm_pdf.evaluate(batch) for batch in utils.batch(x, 15000)]
        # norm_score = [self.norm_pdf.evaluate(batch) for batch in utils.batch(x, 15000)]
        # norm_score = np.array(norm_score)
        # norm_score = norm_score.reshape(np.product(norm_score.shape))
        # norm_score = self.norm_pdf.score_samples(x.reshape(-1, 1))
        norm_score = np.concatenate([self.norm_pdf.evaluate(xi) for xi in utils.batch(x, 1000)])
        t = T(self.norm_pdf)
        t2 = T(self.mal_pdf)

        cores_to_use = math.floor(0.9 * multiprocessing.cpu_count())
        chnk_size = min(round(x.shape[0] / cores_to_use), 200000)
        # with Pool(cores_to_use, initializer=set_niceness, initargs=[19]) as p:
        #     norm_score = p.map(t.predi, x, chunksize=chnk_size)
        # norm_score = np.array(norm_score)
        print('norm done')
        # with Pool(cores_to_use, initializer=set_niceness, initargs=[19]) as p:
        #     mal_score = p.map(t2.predi, x, chunksize=chnk_size)
        # mal_score = np.array(mal_score)

        # with Pool(5) as p:
        # mal_score = p.map(self.mal_pdf.evaluate, x, 3000)
        # mal_score = [self.mal_pdf.evaluate(batch) for batch in utils.batch(x, 15000)]
        # mal_score = np.array(mal_score)

        # mal_score = mal_score.reshape(np.product(mal_score))
        # mal_score = self.mal_pdf.score_samples(x.reshape(-1, 1))
        # mal_score = self.mal_pdf.evaluate(x)
        mal_score = np.concatenate([self.mal_pdf.evaluate(xi) for xi in utils.batch(x, 100000)])
        print('mal done')

        return mal_score > norm_score

    def predict2(self, x, **kwargs):
        norm_score = np.concatenate([self.norm_pdf.evaluate(xi) for xi in utils.batch(x, 1000)])
        mal_score = np.concatenate([self.mal_pdf.evaluate(xi) for xi in utils.batch(x, 100000)])

        return norm_score, mal_score


if __name__ == '__main__':
    model = KDEDownstream()
    model.compile()
    train = np.array([1, 2, 3, 4])
    # y = np.array([0, 0, 1, 1])
    y = np.array([1, 1, 0, 0])
    model.fit(train, y)
    print(str(model.predict(np.array([1.5, 3.5]))))
    import matplotlib.pyplot as plt

    xx = np.linspace(0, 5, 100)
    plt.plot(xx, model.predict(xx))
    plt.show()
