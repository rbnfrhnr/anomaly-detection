from data.loader import load
from data.mit_bih import load as mit_bih_load


def get_loader(data_set):
    if data_set == 'ctu-13':
        return load
    if data_set == 'mit-bih':
        return mit_bih_load
