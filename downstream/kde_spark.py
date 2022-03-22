import math
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.mllib.stat import KernelDensity
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


class KDESparkDownstream(keras.Model):

    def __init__(self, **params):
        super(KDESparkDownstream, self).__init__()
        self.sess = SparkSession.builder.getOrCreate()
        self.sc = self.sess.sparkContext
        self.norm_pdf = KernelDensity()
        self.mal_pdf = KernelDensity()

    def fit(self, x, y, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        # self.mal_pdf = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_mal.reshape(-1, 1))
        x_norm = self.sc.parallelize(x_norm)
        x_mal = self.sc.parallelize(x_mal)
        self.norm_pdf.setSample(x_norm)
        self.mal_pdf.setSample(x_mal)
        return self

    def predict(self, x, **kwargs):
        x = self.sc.parallelize(x)
        return self.mal_pdf.estimate(x) > self.norm_pdf.estimate(x)

    def predict2(self, x, **kwargs):
        norm_score = np.concatenate([self.norm_pdf.evaluate(xi) for xi in utils.batch(x, 1000)])
        mal_score = np.concatenate([self.mal_pdf.evaluate(xi) for xi in utils.batch(x, 100000)])

        return self.mal_pdf.estimate(x), self.norm_pdf.estimate(x)


if __name__ == '__main__':
    model = KDESparkDownstream()
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
