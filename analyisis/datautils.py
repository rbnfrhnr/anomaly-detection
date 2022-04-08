import numpy as np
import math
from glob import glob
from sklearn.preprocessing import MinMaxScaler

def get_data(dataset_nr, window_size):
    file = glob('./' + str(dataset_nr) + '*')[0]
    file_name = file.split('/')[-1]
    data = np.fromfile(file, sep='\n').reshape(-1, 1)
    dataset_name = file_name.split('.')[0]
    dataset_info = dataset_name.split('_')
    train_from = 0
    train_to = int(dataset_info[4])
    test_from = train_to
    anomaly_from = int(dataset_info[5])
    anomaly_to = int(dataset_info[6])
    anomaly_range = np.arange(anomaly_from, anomaly_to)

    train_data = data[:train_to]
    train_data = train_data[:math.floor(train_data.shape[0] / window_size) * window_size]
    train_data_normalized = MinMaxScaler().fit_transform(train_data)
    train_data_normalized = train_data_normalized.reshape(-1, window_size)
    test_data = data[train_to:]
    test_data = test_data[:math.floor(test_data.shape[0] / window_size) * window_size]
    test_data_normalized = MinMaxScaler().fit_transform(test_data)
    test_data_idx = np.arange(test_data.shape[0])
    new_anomaly_from = anomaly_from - train_to
    new_anomaly_to = anomaly_to - train_to
    new_anomaly_range = np.arange(new_anomaly_from, new_anomaly_to)
    test_y = np.isin(test_data_idx, new_anomaly_range).astype(int)
    # will not be relevant for ucr dataset. only here for the kde to fit an anomalicious pdf
    # which will not be used for the identificaiton of anomalies
    dummy_train_mal = np.ones((1, window_size, 1))
    indexer = np.arange(window_size)[None, :] + 1 * np.arange(train_data.reshape(-1).shape[0] - window_size - 1)[:,None]
#     indexer = np.arange(train_data.reshape(-1).shape[0]).reshape(-1, window_size)
    print(indexer)
    train_data_normalized = train_data_normalized.reshape(-1)[indexer]
    train_data_normalized = train_data_normalized.reshape(train_data_normalized.shape[0], window_size)
    
    test_y = test_y.reshape(-1, window_size)
    y_2 = test_y.sum(axis=1)
    y_2 = y_2 >= 1
    y_2 = y_2.reshape(-1)
    
    return train_data_normalized, test_data_normalized.reshape(-1, window_size), y_2

