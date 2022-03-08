from tensorflow import keras
import numpy as np
import tensorflow as tf
import scipy.stats as st
import utils


class BestFitPdfDownstream(keras.Model):

    def __init__(self, **params):
        super(BestFitPdfDownstream, self).__init__()
        self.norm_pdf = None
        self.norm_pdf_name = ''
        self.norm_pdf_params = None
        self.mal_pdf = None
        self.mal_pdf_name = ''
        self.mal_pdf_params = None

    def fit(self, x, y, **kwargs):
        x_norm = x[np.where(y == 0)]
        x_mal = x[np.where(y == 1)]

        normal_dist_n, abparams_norm = utils.best_fit_distribution(x_norm)
        self.norm_pdf = getattr(st, normal_dist_n)
        self.norm_pdf_name = normal_dist_n
        self.norm_pdf_params = abparams_norm

        malicious_dist_n, abparams_mal = utils.best_fit_distribution(x_mal)
        self.mal_pdf = getattr(st, malicious_dist_n)
        self.mal_pdf_name = malicious_dist_n
        self.mal_pdf_params = abparams_mal
        return self

    def predict(self, x, **kwargs):
        norm_score = self.norm_pdf.pdf(x, loc=self.norm_pdf_params[-2], scale=self.norm_pdf_params[-1],
                                       *self.norm_pdf_params[:-2])
        mal_score = self.mal_pdf.pdf(x, loc=self.mal_pdf_params[-2], scale=self.mal_pdf_params[-1],
                                     *self.mal_pdf_params[:-2])
        return mal_score > norm_score

    def predict2(self, x, **kwargs):
        norm_score = self.norm_pdf.pdf(x, loc=self.norm_pdf_params[-2], scale=self.norm_pdf_params[-1],
                                       *self.norm_pdf_params[:-2])
        mal_score = self.mal_pdf.pdf(x, loc=self.mal_pdf_params[-2], scale=self.mal_pdf_params[-1],
                                     *self.mal_pdf_params[:-2])
        return norm_score, mal_score

    def predict_norm(self, x):
        return self.norm_pdf.pdf(x, loc=self.norm_pdf_params[-2], scale=self.norm_pdf_params[-1],
                                 *self.norm_pdf_params[:-2])

    def predict_mal(self, x):
        return self.mal_pdf.pdf(x, loc=self.mal_pdf_params[-2], scale=self.mal_pdf_params[-1],
                                *self.mal_pdf_params[:-2])


if __name__ == '__main__':
    model = BestFitPdfDownstream()
    model.compile()
    train = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 1, 1])
    model.fit(train, y)
    print(str(model.predict(np.array([1.5, 3.5]))))
