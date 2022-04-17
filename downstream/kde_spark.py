import math
import os

import numpy as np
import pandas as pd
from pyspark.mllib.stat import KernelDensity
from pyspark.sql import SparkSession
from tensorflow import keras

import utils


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class KDESparkDownstream(keras.Model):

    def __init__(self, **params):
        super(KDESparkDownstream, self).__init__()
        self.databrick_cfg = params['downstream']['tasks']['kde-spark']
        self.run_dir = params['run-dir']
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
        norms, mals = self.predict2(x, **kwargs)
        return mals > norms

    def predict2(self, x, **kwargs):
        # x1 = self.sc.parallelize(x)
        # x2 = self.sc.parallelize(x.copy)
        size = math.floor(x.shape[0] / 10)
        batch_nr = 0
        mals = []
        norms = []
        for xx in batch(x, size):
            mal = self.mal_pdf.estimate(xx.copy().astype(float).tolist())
            norm = self.norm_pdf.estimate(xx.copy().astype(float).tolist())
            mals.append(mal)
            norms.append(norm)
            mal_rec = pd.DataFrame(data=np.stack([xx.reshape(-1), mal.reshape(-1)]).T,
                                   columns=['recon-error', 'kde-prediction'])
            norm_rec = pd.DataFrame(data=np.stack([xx.reshape(-1), norm.reshape(-1)]).T,
                                    columns=['recon-error', 'kde-prediction'])
            mal_rec.to_csv(self.run_dir + '/spark-kde-mal-batch-' + str(batch_nr) + '.csv')
            norm_rec.to_csv(self.run_dir + '/spark-kde-norm-batch-' + str(batch_nr) + '.csv')
            batch_nr += 1
        mals = np.concatenate(mals, axis=0)
        norms = np.concatenate(norms, axis=0)
        return norms, mals


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
