import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def vae_loss(reconstructed_x, x, mean, logvar, model, lens, device, step, x0, kld_w):
    if model == "RNN":
        # mask = _sequence_mask(lens).to(device)
        # # nb_tokens = int(torch.sum(mask).data[0])
        # BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='none').sum(2)
        # BCE = BCE * mask
        # KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),axis=1)
        # total = BCE.sum(1) + KLD
        mask = _sequence_mask(lens).to(device)
        # nb_tokens = int(torch.sum(mask).data[0])
        BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='none').sum(2)
        BCE = BCE * mask
        KLD = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp(),axis=1)
        # DAVE
        # alpha = 1e-8
        # KLD = -0.5 * torch.mean(alpha - alpha * math.log(alpha)  + alpha * logvar - mean.pow(2) - logvar.exp(),axis=1)
        if kld_w:
            kl_weight = kl_anneal_function(step, x0)
        else:
            kl_weight = 1
        total = BCE.mean(1) + kl_weight * KLD # sum up whole sequence, sum of binary cross entropy of whole sequence and one kld
    else:
        # mean
        # BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='none').mean(1)
        # KLD = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp(),axis=1)
        # test - sum
        BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='none').sum(1)
        alpha = 1e-10
        # KLD = -0.5 * torch.mean(alpha - alpha * math.log(alpha)  + alpha * logvar - mean.pow(2) - logvar.exp(),axis=1)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),axis=1)
        total = BCE + KLD
    return total, BCE, KLD

def inference_loss(reconstructed_x, x, lens, device):
    mask = _sequence_mask(lens).to(device)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='none').sum(2)
    BCE = BCE * mask
    return BCE

def kl_anneal_function(step, x0):
    return min(1, step/x0)
    # if anneal_function == 'logistic':
    #     return float(1/(1+np.exp(-k*(step-x0))))
    # elif anneal_function == 'linear':
    #     return min(1, step/x0)
    
    
def _sequence_mask(sequence_length, max_len=None):
    max_len = sequence_length.data.max().item()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand