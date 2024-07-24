import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import *
from dataset import *
from utils import *
from transformer import Transformer


class Feed(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Feed, self).__init__()
        self.fc6 = nn.Linear(X_dim, 1024)
        self.fc6_bn = nn.BatchNorm1d(1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc7_bn = nn.BatchNorm1d(2048)
        self.fc8 = nn.Linear(2048, 2048)
        self.fc8_bn = nn.BatchNorm1d(2048)
        self.fc9 = nn.Linear(2048, gene_number)

    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        if relu:
            return F.relu(self.fc9(h8))
        else:
            return self.fc9(h8)


class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, nhid, out_features, dropout, alpha, heads=4):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj).squeeze(0) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)



class HistoSGE(nn.Module):
    def __init__(self, in_features, depth, heads, n_genes=1000, dropout=0.):
        super(HistoSGE, self).__init__()
        self.x_embed = nn.Embedding(512, in_features)
        self.y_embed = nn.Embedding(512, in_features)
        self.trans = Transformer(dim=in_features, depth=depth, heads=heads, dim_head=64, mlp_dim=in_features,
                                 dropout=dropout)

        self.gene_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, n_genes)
        )
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x, centers):
        centers_x = self.x_embed(centers[:, :, 0].long())
        centers_y = self.y_embed(centers[:, :, 1].long())

        x = x + centers_x + centers_y
        h = self.trans(x)
        x = self.gene_head(h)

        return h, F.relu(x)
