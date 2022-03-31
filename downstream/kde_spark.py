import os

import numpy as np
from pyspark.mllib.stat import KernelDensity
from pyspark.sql import SparkSession
from tensorflow import keras

import utils


class KDESparkDownstream(keras.Model):

    def __init__(self, **params):
        super(KDESparkDownstream, self).__init__()
        self.databrick_cfg = params['downstream']['tasks']['kde-spark']
        os.environ['DEBUG_IGNORE_VERSION_MISMATCH'] = str(self.databrick_cfg['ignore-version-mismatch'])
        os.environ['PYSPARK_PYTHON'] = "/usr/local/bin/python3.8"
        os.environ['PYSPARK_DRIVER_PYTHON'] = "/usr/local/bin/python3.8"
        # os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/robin/Documents/lbnl/crd/anomaly-detection/venv/bin/python3.9'
        # export
        # PYSPARK_DRIVER_PYTHON = ipython3
        self.sess = SparkSession.builder \
            .config("spark.databricks.service.address", self.databrick_cfg['host']) \
            .config("spark.databricks.service.clusterId", self.databrick_cfg['cluster']) \
            .config("spark.databricks.service.token", self.databrick_cfg['token']) \
            .config("spark.databricks.service.port", self.databrick_cfg['port']) \
            .getOrCreate()
        # self.sess = SparkSession.builder.getOrCreate()
        self.sc = self.sess.sparkContext
        self.norm_pdf = KernelDensity()
        self.mal_pdf = KernelDensity()

    def fit(self, x, y, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        # self.mal_pdf = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_mal.reshape(-1, 1))
        x_norm = self.sc.parallelize(x_norm.astype(float).tolist())
        x_mal = self.sc.parallelize(x_mal.astype(float).tolist())
        self.norm_pdf.setSample(x_norm)
        self.mal_pdf.setSample(x_mal)
        return self

    def predict(self, x, **kwargs):
        mal = self.mal_pdf.estimate(x.copy().astype(float).tolist())
        norm = self.norm_pdf.estimate(x.copy().astype(float).tolist())
        return mal > norm

    def predict2(self, x, **kwargs):
        # x1 = self.sc.parallelize(x)
        # x2 = self.sc.parallelize(x.copy)
        mal = self.mal_pdf.estimate(x.copy().astype(float).tolist())
        norm = self.norm_pdf.estimate(x.copy().astype(float).tolist())
        return norm, mal


if __name__ == '__main__':
    config = utils.read_cfg('../config/mit-bih/template-rvae-bih.yaml')
    model = KDESparkDownstream(**config)
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
