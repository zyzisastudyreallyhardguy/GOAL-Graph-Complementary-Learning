from torch_geometric.nn import GATConv, GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import dgl.function as fn

import torch

from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
)

from torch_geometric.nn.inits import reset
from torch.nn.functional import normalize
# from ..inits import reset

'''GNN Encoder Trianing'''
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_lay):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels, heads, dropout=0.6))
        for i in range(num_lay - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels * heads, dropout=0.6))
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x ,edge_index)
            if i == len(self.layers) - 1:
                embed = copy.copy(x)
        return x, embed

'''GNN Encoder Training'''
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout = 0.5):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == len(self.convs) - 2:
                embed = copy.copy(x)
        x = self.convs[-1](x, adj_t)
        return x, embed

class MPConv(nn.Module):
    def __init__(self):
        super(MPConv, self).__init__()

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        # normalization by src node
        feat = feat * norm
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))

        feat = graph.ndata['h']
        # normalization by dst node
        feat = feat * norm
        return feat

class GNN_high(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_lay):
        super(GNN_high, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size, bias = False))
        for i in range(num_lay - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias = False))
        self.layers.append(nn.Linear(hidden_size, out_size, bias = False))
        self.mp = MPConv()
        self.dropout = nn.Dropout(0.6)

    def forward(self, feat, graph, gamma = 0.3):
        h = feat
        for i, layer in enumerate(self.layers):
            temp_h = copy.copy(h)
            h = F.normalize(self.mp(graph, h))
            h = - layer(h)
            h = F.normalize(h) + gamma * layer(temp_h)
            if i == len(self.layers) - 1:
                embed = copy.copy(h)
        return h, embed
    
EPS = 1e-15
MAX_LOGSTD = 10

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        # else:
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        # logsig = nn.LogSigmoid()
        return torch.sigmoid(value) if sigmoid else value

        # return logsig(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
class ReconstructEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        ReconstructEncoder.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        #prediction accuracy
        # print('Prediction Accuracy:')
        # print((y == (pred > 0.5).astype('int')).sum() / y.shape[0])
        return roc_auc_score(y, pred), average_precision_score(y, pred)


def listNet(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

class MPConv(nn.Module):
    def __init__(self):
        super(MPConv, self).__init__()

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)
        feat = feat * norm
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        feat = graph.ndata['h']
        feat = feat * norm
        return feat


class GOAL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer, dropout=0.5):
        super().__init__()
        self.mp = MPConv()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()  # BatchNorm layers
        self.dropout = nn.Dropout(dropout)
        self.layers.append(nn.Linear(in_size, hid_size, bias=False))
        self.bns.append(nn.BatchNorm1d(hid_size))
        for i in range(num_layer - 2):
            self.layers.append(nn.Linear(hid_size, hid_size, bias=False))
            self.bns.append(nn.BatchNorm1d(hid_size))
        self.layers.append(nn.Linear(hid_size, out_size, bias=False))

    def forward(self, g, g_high, features, alpha, beta, gamma, delta):
        h = features
        h = F.normalize(h)
        for i, layer in enumerate(self.layers):
            temp_h = copy.copy(h)
            h_p = self.mp(g, h)
            h_n = F.normalize(self.mp(g_high, h))
            h_p = self.layers[i](h_p)
            h_n = -self.layers[i](h_n)
            h_i = self.layers[i](temp_h)
            h = beta * h_p + gamma * F.normalize(h_n) + alpha * h_i - delta * F.normalize(self.mp(g_high, h_p))
            if i < 2:
                h = self.bns[i](h)  # Apply BatchNorm
                h = F.relu(h)
                h = self.dropout(h)  # Apply Dropout
        return h
