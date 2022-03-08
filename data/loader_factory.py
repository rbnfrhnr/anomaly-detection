from data.loader import load
from data.mit_bih import fetch_sliced


def get_loader(data_set):
    if data_set == 'ctu-13':
        return load
    if data_set == 'mit-bih':
        return fetch_sliced
