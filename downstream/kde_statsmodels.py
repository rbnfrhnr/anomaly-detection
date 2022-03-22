from datetime import datetime

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from tensorflow import keras
import matplotlib.pyplot as plt
import utils


def plot_density(kde_model, x_norm, x_mal):
    x_min = min(np.min(x_norm), np.min(x_mal))
    x_max = max(np.max(x_norm), np.max(x_mal))
    xx = np.linspace(x_min, x_max, 200)

    plt.plot(xx, kde_model.norm_pdf.evaluate(xx))
    plt.plot(xx, kde_model.mal_pdf.evaluate(xx))
    plt.hist(x_norm, bins=100, density=True)
    plt.hist(x_mal, bins=100, density=True)
    d = datetime.today()
    suffix = d.strftime("%Y-%m-%d-%H-%M-%S")
    plt.show()


class KDEStatsDownstream(keras.Model):

    def __init__(self, **params):
        super(KDEStatsDownstream, self).__init__()
        self.norm_pdf = None
        self.mal_pdf = None

    def fit(self, x, y, plot_hist=True, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        self.norm_pdf = KDEUnivariate(x_norm)
        self.norm_pdf.fit()

        self.mal_pdf = KDEUnivariate(x_mal)
        self.mal_pdf.fit()

        if plot_hist:
            plot_density(self, x_norm, x_mal)
        return self

    def predict(self, x, batch_size=None, **kwargs):
        batch_size = x.shape[0]
        norm_score = np.concatenate([self.norm_pdf.evaluate(xi) for xi in utils.batch(x, batch_size)])
        mal_score = np.concatenate([self.mal_pdf.evaluate(xi) for xi in utils.batch(x, batch_size)])
        return mal_score > norm_score

    def predict2(self, x, batch_size=None, **kwargs):
        batch_size = x.shape[0]
        norm_score = np.concatenate([self.norm_pdf.evaluate(xi) for xi in utils.batch(x, batch_size)])
        mal_score = np.concatenate([self.mal_pdf.evaluate(xi) for xi in utils.batch(x, batch_size)])

        return norm_score, mal_score


if __name__ == '__main__':
    model = KDEStatsDownstream()
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
