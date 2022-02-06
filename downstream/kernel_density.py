from tensorflow import keras
import numpy as np
import tensorflow as tf
import scipy.stats as st
import utils
from statsmodels.nonparametric.kde import KDEUnivariate


class KDEDownstream(keras.Model):

    def __init__(self, **params):
        super(KDEDownstream, self).__init__()
        self.norm_pdf = None
        self.mal_pdf = None

    def fit(self, x, y, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        self.norm_pdf = KDEUnivariate(x_norm)
        self.norm_pdf.fit()

        self.mal_pdf = KDEUnivariate(x_mal)
        self.mal_pdf.fit()

        return self

    def predict(self, x, **kwargs):
        norm_score = self.norm_pdf.evaluate(x)
        mal_score = self.mal_pdf.evaluate(x)
        return mal_score > norm_score


if __name__ == '__main__':
    model = KDEDownstream()
    model.compile()
    train = np.array([1, 2, 3, 4])
    # y = np.array([0, 0, 1, 1])
    y = np.array([1, 1, 0, 0])
    model.fit(train, y)
    print(str(model.predict(np.array([1.5, 3.5]))))
