import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, step=2., droprate=0.3):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.droprate = droprate
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        tmp_dim = hidden_dim
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(tmp_dim, output_dim))
            else:
                tmp_hidden_dim = tmp_dim
                tmp_dim = int(tmp_dim / step)
                if tmp_dim < output_dim:
                    tmp_dim = tmp_hidden_dim
                self.enc.append(nn.Linear(tmp_hidden_dim, tmp_dim))
        # self.init_weights()

    def forward(self, x):
        z = self.encode(x)
        return z

    def encode(self, x):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if self.droprate:
                    h = torch.dropout(h, self.droprate, train=self.training)
                h = layer(h)
            else:
                if self.droprate:
                    h = torch.dropout(h, self.droprate, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)


class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
        ])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.enc.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.enc.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h


class GraphEnc(nn.Module):
    def __init__(self, order=2):
        super(GraphEnc, self).__init__()
        self.order = order

    def forward(self, x, adj, order=None):
        if order is not None:
            self.order = order
        adj = self.normalize_adj(adj)
        z = self.message_passing_global(x, adj)
        return z

    def message_passing_global(self, x, adj):
        h = x
        for i in range(self.order):
            h = torch.matmul(adj, h) + (1 * x)
        return h

    def normalize_adj(self, x):
        D = x.sum(1).detach().clone()
        r_inv = D.pow(-1).flatten()
        r_inv = r_inv.reshape((x.shape[0], -1))
        r_inv[torch.isinf(r_inv)] = 0.
        x = x * r_inv
        return x


class SimInfoExtror(nn.Module):
    def __init__(self, input_dim_x, hidden_dim_x, output_dim_x, input_dim_a, hidden_dim_a, output_dim_a,
                 class_num, num_layers_x=2, step_x=2, num_layers_a=2, step_a=2, k=10):
        super(SimInfoExtror, self).__init__()
        self.k = k
        self.x_extr = MLP(input_dim_x, hidden_dim_x, output_dim_x, num_layers_x, step_x)
        self.a_extr = MLP(input_dim_a, hidden_dim_a, output_dim_a, num_layers_a, step_a)
        self.cluster_layer_x = Parameter(torch.Tensor(class_num, output_dim_x))
        self.register_parameter('centroid_x', self.cluster_layer_x)
        self.cluster_layer_a = Parameter(torch.Tensor(class_num, output_dim_a))
        self.register_parameter('centroid_a', self.cluster_layer_a)

    def forward(self, x, adj, weights_a):
        zx = self.x_extr(x)
        zx_norm = F.normalize(zx, p=2, dim=-1)
        homo_x = torch.mm(zx, zx.T)

        za = self.a_extr(adj)
        za_norm = F.normalize(za, p=2, dim=-1)
        homo_a = torch.mm(za, za.T)

        homo_x_norm = F.normalize(homo_x, p=2, dim=-1)
        homo_a_norm = F.normalize(homo_a, p=2, dim=-1)
        S = (weights_a[0] * homo_x_norm + weights_a[1] * homo_a_norm) / sum(weights_a)

        _, indices = torch.topk(S, k=self.k, dim=-1)
        S_dis = torch.zeros_like(S)
        S_dis[torch.arange(S.shape[0]).reshape(-1, 1), indices] = 1
        S = S_dis + torch.eye(S.shape[0], device=S.device)

        return zx_norm, homo_x, za_norm, homo_a, S

    def predict_distribution(self, z, layer, alpha=1.0):
        c = layer
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


class EnDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim):
        super(EnDecoder, self).__init__()

        self.enc = LatentMappingLayer(feat_dim, hidden_dim, latent_dim, num_layers=2)
        self.dec_f = LatentMappingLayer(latent_dim, hidden_dim, feat_dim, num_layers=2)

    def forward(self, x, dropout=0.1):
        z = self.enc(x, dropout)
        z_norm = F.normalize(z, p=2, dim=1)
        x_pred = torch.sigmoid(self.dec_f(z_norm, dropout))
        return x_pred, z_norm


class SMHGC(nn.Module):
    def __init__(self,
                 input_dim_x, hidden_dim_x, output_dim_x,
                 input_dim_a, hidden_dim_a, output_dim_a,
                 input_dim_g, hidden_dim_g, output_dim_g,
                 class_num, node_num, view_num, order=2, num_layers_x=2, step_x=2, num_layers_a=4, step_a=2, k=10):
        super(SMHGC, self).__init__()

        self.class_num = class_num
        self.node_num = node_num
        self.view_num = view_num
        self.order = order

        self.homo_extrs = nn.ModuleList([
            SimInfoExtror(input_dim_x, hidden_dim_x, output_dim_x, input_dim_a, hidden_dim_a, output_dim_a,
                          class_num, num_layers_x, step_x, num_layers_a, step_a, k=k) for _ in range(view_num)
        ])

        self.graphencs = nn.ModuleList([
            GraphEnc(order=order) for _ in range(view_num)
        ])


        self.endecs = nn.ModuleList([
            EnDecoder(input_dim_g, hidden_dim_g, output_dim_g) for _ in range(view_num)
        ])

        self.cluster_layer = [Parameter(torch.Tensor(class_num, output_dim_g)) for _ in range(view_num)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, output_dim_g)))
        for i in range(view_num):
            self.register_parameter('centroid_{}'.format(i), self.cluster_layer[i])
        self.register_parameter('centroid_{}'.format(view_num), self.cluster_layer[view_num])

    def forward(self, xs, adjs, weights_a, weights_h, order=None):
        if order is not None:
            self.order = order

        zx_norms = []
        homo_xs = []
        za_norms = []
        homo_as = []
        hs = []
        qgs = []
        x_preds = []
        Ss = []

        for v in range(self.view_num):
            zx_norm, homo_x, za_norm, homo_a, S = self.homo_extrs[v](xs[v], adjs[v], weights_a[v])
            zx_norms.append(zx_norm)
            homo_xs.append(homo_x)
            za_norms.append(za_norm)
            homo_as.append(homo_a)

            x_pred, z_norm = self.endecs[v](xs[v])
            x_preds.append(x_pred)
            h = self.graphencs[v](z_norm, S)
            h = F.normalize(h, p=2, dim=-1)
            hs.append(h)

            qg = self.predict_distribution(h, v)
            qgs.append(qg)
            Ss.append(S)

        h_all = sum(weights_h[v] * hs[v] for v in range(self.view_num)) / sum(weights_h)
        qg = self.predict_distribution(h_all, -1)
        qgs.append(qg)
        return zx_norms, homo_xs, za_norms, homo_as, hs, h_all, qgs, x_preds, Ss

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

