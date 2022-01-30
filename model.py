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
from torch.autograd import Function
# from sklearn.metrics import confusion_matrix, precision_recall_curve
# from sklearn.model_selection import train_test_split

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        input = input.view(self.seq_len, -1, self.input_size) # seq_len, batch_size, feature_size
        output, _ = self.rnn(input)
        output = output.squeeze()
        output = self.linear(output)
        output = self.softmax(output)
        return output[-1]

class DNN(nn.Module):
    def __init__(self, dims, activation):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(dims[:-1], dims[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        self.softmax = nn.LogSoftmax(dim=-1)
        self.acti = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activations[self.acti](x)
            else:
                result = x
        output = self.softmax(result)
        return output.squeeze()
    
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, seq_len):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.view(-1, self.seq_len, self.input_size)
        encoded_input, hidden = self.lstm(inputs)
        encoded_input = self.relu(encoded_input)
        return encoded_input

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, seq_len):
        super(LSTM_AE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, batch_size, seq_len)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)
        
    def forward(self, inputs):
        encoded_input = self.encoder(inputs)
        decoded_output = self.decoder(encoded_input)
        return encoded_input, decoded_output

class EncoderDNN(nn.Module):
    def __init__(self, dims, activation):
        super(EncoderDNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(dims[:-1], dims[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.acti = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activations[self.acti](x)
        return x

class DecoderDNN(nn.Module):
    def __init__(self, dims, activation):
        super(DecoderDNN, self).__init__()
        dims_re = list(reversed(dims))
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(dims_re[:-1], dims_re[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.acti = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.sigmoid(x)
        return x

class DNN_AE(nn.Module):
    def __init__(self, dims, activation):
        super(DNN_AE, self).__init__()
        self.encoder = EncoderDNN(dims, activation)
        self.decoder = DecoderDNN(dims, activation)
        
    def forward(self, inputs):
        encoded_input = self.encoder(inputs)
        decoded_output = self.decoder(encoded_input)
        return encoded_input, decoded_output
    
class EncoderDNN_vae(nn.Module):
    def __init__(self, dims, activation):
        super(EncoderDNN_vae, self).__init__()
        d = dims.copy()
        self.latent_size = d[-1]
        d[-1] = d[-1] * 2
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(d[:-1], d[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.acti = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activations[self.acti](x)
            else:
                gaussian_params = x
            # print("encoder", x)
        mu = gaussian_params[:,:self.latent_size]
        logvar = gaussian_params[:, self.latent_size:]
        return mu, logvar

class DecoderDNN_vae(nn.Module):
    def __init__(self, dims, activation):
        super(DecoderDNN_vae, self).__init__()
        dims_re = list(reversed(dims))
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(dims_re[:-1], dims_re[1:])])
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()]
        ])
        # self.softmax = nn.LogSoftmax(dim=-1)
        self.acti = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # print('decode', x)
            x = layer(x)
            x = torch.sigmoid(x)
        return x
    
class DNN_VAE(nn.Module):
    def __init__(self, dims, activation):
        super(DNN_VAE, self).__init__()
        self.encoder = EncoderDNN_vae(dims, activation)
        self.decoder = DecoderDNN_vae(dims, activation)
            
    def forward(self, inputs):
        mu, logvar = self.encoder(inputs)
        z = self.reparam(mu, logvar)
        decoded_output = self.decoder(z)
        return mu, logvar, decoded_output, z
    
    def reparam(self, mu, logvar):
        if self.training:
            # z = torch.randn_like(sigma, dtype=torch.float32) * sigma + mu
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

class EncoderRNN_vae(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, latent_size, bidirectional):
        super(EncoderRNN_vae, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.hidden2mean = nn.Linear(hidden_size*num_layers*2, latent_size)
            self.hidden2logv= nn.Linear(hidden_size*num_layers*2, latent_size)
        else:
            self.hidden2mean = nn.Linear(hidden_size*num_layers, latent_size)
            self.hidden2logv= nn.Linear(hidden_size*num_layers, latent_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # inputs = inputs.view(-1, self.seq_len, self.input_size)
        _, hidden = self.gru(inputs)
        hidden = self.relu(hidden)
        if self.bidirectional:
            hidden = hidden.view(-1, self.num_layers*self.hidden_size*2)
        else:
            hidden = hidden.view(-1, self.num_layers*self.hidden_size) 
        mu = self.hidden2mean(hidden)
        logvar = self.hidden2logv(hidden)
        return mu, logvar

class DecoderRNN_vae(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, bidirectional):
        super(DecoderRNN_vae, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.latent2hidden = nn.Linear(latent_size, hidden_size * num_layers*2)
        else:
            self.latent2hidden = nn.Linear(latent_size, hidden_size * num_layers)
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, input_size)
            # self.linear_1 = nn.Linear(hidden_size * 2, hidden_size)
            # self.linear_2 = nn.Linear(hidden_size, input_size) 
        else:
            self.linear = nn.Linear(hidden_size, input_size)
            # self.linear_1 = nn.Linear(hidden_size, int(hidden_size/2))
            # self.linear_2 = nn.Linear(int(hidden_size/2), input_size) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, z):
        hidden = self.latent2hidden(z)
        if self.bidirectional:
            hidden = hidden.view(self.num_layers*2, -1, self.hidden_size)
        else:
            hidden = hidden.view(self.num_layers, -1, self.hidden_size)
        # non-linear
        hidden = self.relu(hidden)
        decoded_output, _ = self.gru(inputs, hidden)
        decoded_output, _ = pad_packed_sequence(decoded_output, batch_first=True)
        decoded_output = self.sigmoid(self.linear(decoded_output.data))
        # decoded_output = self.sigmoid(self.linear_2(self.sigmoid(self.linear_1(decoded_output.data))))
        return decoded_output
    

class RNN_VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, latent_size, bidirectional):
        super(RNN_VAE, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.encoder = EncoderRNN_vae(input_size, hidden_size, num_layers, batch_size, latent_size, bidirectional)
        self.decoder = DecoderRNN_vae(input_size, hidden_size, num_layers, latent_size, bidirectional)
        
    def forward(self, inputs_encoder):
        mu, logvar = self.encoder(inputs_encoder)
        z = self.reparam(mu, logvar)
        padded, lens = pad_packed_sequence(inputs_encoder, batch_first=True)
        decoded_input = padded[:,1:,:] #to make rnn decoder input
        m = nn.ConstantPad2d((0, 0, 1, 0), 0.0)
        inputs_decoder = m(decoded_input)
        inputs_decoder = pack_padded_sequence(inputs_decoder, lens, batch_first=True)
        decoded_output = self.decoder(inputs_decoder, z)
        # return mu.squeeze(0), logvar.squeeze(0), decoded_output, padded, lens, z
        return mu, logvar, decoded_output, padded, lens, z

    def reparam(self, mu, logvar):
        if self.training:
            # z = torch.randn_like(sigma, dtype=torch.float32) * sigma + mu
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output

class DANN_Classifier(nn.Module):
    def __init__(self, input_size):
        super(DANN_Classifier, self).__init__()
        self.Linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, recon):
        recon = ReverseLayerF.apply(recon)
        result = self.sigmoid(self.Linear(recon))
        return result