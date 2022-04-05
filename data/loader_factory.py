from data.loader import load
from data.mit_bih import load_windowed_no_overlap, load_sliding_window
from data.ucr_loader import load as ucr_loader


def get_loader(data_set):
    if data_set == 'ctu-13':
        return load
    if data_set == 'mit-bih':
        return load_windowed_no_overlap
        # return load_sliding_window
    if data_set == 'ucr':
        return ucr_loader
