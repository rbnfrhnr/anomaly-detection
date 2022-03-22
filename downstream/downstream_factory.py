from downstream import best_fit_pdf
from downstream import kde_statsmodels
from downstream import kde_sklearn
from downstream import kde_spark


def get_downstream_task(name, **params):
    if name == 'best-fit-pdf':
        return best_fit_pdf.BestFitPdfDownstream(**params)
    if name == 'kde':
        return kde_statsmodels.KDEStatsDownstream(**params)
    if name == 'kde-sklearn':
        return kde_sklearn.KDESklearnDownstream(**params)
    if name == 'kde-spark':
        return kde_spark.KDESparkDownstream(**params)
    return None
