from downstream import best_fit_pdf
from downstream import kernel_density


def get_downstream_task(name, **params):
    if name == 'best-fit-pdf':
        return best_fit_pdf.BestFitPdfDownstream(**params)
    if name == 'kde':
        return kernel_density.KDEDownstream(**params)
    return None
