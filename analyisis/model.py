import torch
import numpy as np
import math


class SubsequenceRNN(torch.nn.Module):
    """
       d_in = (time_window_size, feature_dim)
       d_out = (...)
       subsequences = [
           len_1,
           len_2,
           ...
       ]
    """

    def __init__(self, d_in, d_out, ranges=None, n_subsequences=2, subsequences=None):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super(SubsequenceRNN, self).__init__()
        self.device = 'cuda'
        if ranges is None:
            ranges = (2, math.floor(d_in / 2))
        if subsequences is None:
            subsequences = np.random.choice(ranges[1] - ranges[0], n_subsequences) + ranges[0]
        print(ranges)
        print(subsequences)
        self.d_in = d_in
        self.d_out = d_out
        self.ranges = ranges
        self.n_subseqences = n_subsequences
        self.subsequences = subsequences
        print(d_in)
        self.sub_input_dims = [(math.floor(d_in / sub), sub) for sub in subsequences]
        print(self.sub_input_dims)
        self.rnns = [torch.nn.GRU(subdim[1], 10, 2, batch_first=True, dropout=0.1) for subdim in self.sub_input_dims]

        print(self.rnns)
        linear_in_dim = sum([gru.hidden_size * self.sub_input_dims[idx][0] for idx, gru in enumerate(self.rnns)])
        self.linear_layer = torch.nn.Linear(linear_in_dim, d_out)
        print(self.linear_layer)

        self.recon_layer = torch.nn.Linear(d_out, d_in)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #         xx = x.reshape(-1)
        #         print(xx.shape)
        x.to(self.device)
        y = torch.zeros(x.shape[0], 0).to(self.device)
        y.to(self.device)
        for rnn in self.rnns:
            xx = x[:, :math.floor(x.shape[1] / rnn.input_size) * rnn.input_size:, ]
            xx = xx.reshape(x.shape[0], -1, rnn.input_size)
            xx.to(self.device)
            #             print(xx.shape)
            xx.to(self.device)
            yy, yh = rnn(xx)
            yy.to(self.device)
            yh.to(self.device)
            #             print(yy.shape)
            # print(self.device, y.is_cuda, yy.is_cuda)
            y = torch.cat([y, yy.flatten(1)], axis=1)
            y.to(self.device)
        #         print(y.shape)
        y.to(self.device)
        y_n = self.linear_layer(y)
        #         print(y_n.shape)
        y_recon = self.recon_layer(y_n)
        return y_recon

    def to(self, device):
        self.device = device
        self.rnns = [rnn.to(self.device) for rnn in self.rnns]
        self.linear_layer.to(self.device)
        self.recon_layer.to(self.device)
        super().to(device)

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'
