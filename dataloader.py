import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
import time
import random
import math
import torch
import numpy as np


class LSTM_VAE_dataloader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, sampler=None, shuffle=False):
        super(LSTM_VAE_dataloader, self).__init__(
            dataset = dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = self._collate_fn,
            shuffle = shuffle
        )
    def _collate_fn(self, list_inputs):
        # sorting sequences by length
        encode_input = [list_input[0] for list_input in list_inputs]
        order = np.argsort([item.shape[0] for item in encode_input])
        list_sorted_e = [list_inputs[i][0] for i in order[::-1]]
        list_sorted_y = [list_inputs[i][1] for i in order[::-1]]
        flatten_y = [y for y in list_sorted_y]
        return pack_sequence(list_sorted_e), flatten_y, order[::-1]

    