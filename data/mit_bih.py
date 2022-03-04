import wfdb
import numpy as np
import math


def preprocess(data, time_steps=5, **kwargs):
    p_data = np.array([rec[0: math.floor(rec.shape[0] / time_steps) * time_steps] for rec in data])
    p_data = np.concatenate([rec.reshape(math.floor(rec.shape[0] / time_steps), time_steps, 1) for rec in p_data])
    return p_data


def load_multiple(data_location, sets=[802, 802], **kwargs):
    sets = np.concatenate([load_mit_bih(data_location, nr=set_nr, **kwargs) for set_nr in sets])
    norm = np.concatenate(sets[0::2])
    mal = np.concatenate(sets[1::2])
    return norm, mal


def load_mit_bih(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr)
    data_file = data_location + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    norm = []
    mal = []
    prev = 0
    for i in range(0, len(anot.symbol)):
        if anot.symbol[i] != 'N':
            mal.append(sigs[0][prev:anot.sample[i]])
            mal.append(sigs[1][prev:anot.sample[i]])
        else:
            norm.append(sigs[0][prev:anot.sample[i]])
            norm.append(sigs[1][prev:anot.sample[i]])
        prev = anot.sample[i]
    norm = np.array(norm)
    mal = np.array(mal)
    return np.array([norm, mal])

def load_mit_bih2(data_location, nr=802, **kwargs):
    anot_file = data_location + str(nr)
    data_file = data_location + str(nr)
    anot = wfdb.rdann(anot_file, 'atr')
    data = wfdb.rdsamp(data_file)
    mal_bound = round(anot.fs / 10)
    sigs = np.array([data[0][:, 0], data[0][:, 1]])
    mal_idx = np.argwhere(np.array(anot.symbol) != 'N')
    mal_ranges = np.array([np.array([anot.sample[mal][0]-mal_bound, (
            anot.sample[mal] + np.round((anot.sample[mal_idx[idx] + 1][0] - anot.sample[mal]) / 2)).astype(int)[0]])
                           for idx, mal in enumerate(mal_idx)])
    mal_ranges_values = np.array([np.array(list(range(interval[0], interval[1]))) for interval in mal_ranges])
    mal_ex = np.array([sigs[0][r] for r in mal_ranges_values])
    xx = range(0, mal_ex[0].shape[0])
    plt.plot(xx, mal_ex[0])
    plt.show()

if __name__ == '__main__':
    # load_mit_bih('C:\workarea\mit-bih\\')
    norm, mal = load_multiple('C:\workarea\mit-bih\\')
    preprocess(norm)
