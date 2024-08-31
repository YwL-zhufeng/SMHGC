import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from utils import load_data, cal_homo_ratio, rw_normalize, minmaxnormalize, normalize_weight
from model import SMHGC
from evaluation import eva
import pandas as pd
from settings import get_settings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='texas', help='datasets: acm, dblp, texas, chameleon, acm00, acm01, acm02, acm03, acm04, acm05')
parser.add_argument('--train', type=int, default=0, help='training mode, 1 or 0')
parser.add_argument('--cuda_device', type=int, default=0, help='')
parser.add_argument('--use_cuda', type=bool, default=True, help='')
args = parser.parse_args()

train = (args.train == 1)
dataset = args.dataset
use_cuda = args.use_cuda
cuda_device = args.cuda_device

settings = get_settings(parser, dataset)
path = settings.path
order = settings.order
weight_soft_h = settings.weight_soft_h
weight_soft_S = settings.weight_soft_S
k_for_disc = settings.K

hidden_dim_x = settings.hidden_dim_x
output_dim_x = settings.output_dim_x
hidden_dim_a = settings.hidden_dim_a
output_dim_a = settings.output_dim_a
hidden_dim_g = settings.hidden_dim_g
output_dim_g = settings.output_dim_g
num_layers_x = settings.num_layers_x
num_layers_a = settings.num_layers_a

epoch = settings.epoch
patience = settings.patience
lr = settings.lr
weight_decay = settings.weight_decay

update_interval = settings.update_interval
random_seed = settings.random_seed

torch.manual_seed(random_seed)


# load dataset
labels, adjs, adjs_labels, shared_feature, shared_feature_label, num_graph = load_data(dataset, path)

# print dataset info
for v in range(num_graph):
    r, homo = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
    print(r, homo)
print('nodes: {}'.format(shared_feature_label.shape[0]))
print('features: {}'.format(shared_feature_label.shape[1]))
print('class: {}'.format(labels.max() + 1))

feat_dim = shared_feature.shape[1]
class_num = labels.max().item() + 1
y = labels.cpu().numpy()
node_num = shared_feature.shape[0]

xs = []
as_input = []
for v in range(num_graph):
    xs.append(shared_feature_label)

model = SMHGC(feat_dim, hidden_dim_x, output_dim_x,
              node_num, hidden_dim_a, output_dim_a,
              feat_dim, hidden_dim_g, output_dim_g,
              class_num, node_num, num_graph, order=order, k=k_for_disc,
              num_layers_x=num_layers_x, num_layers_a=num_layers_a)

if use_cuda:
    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    model = model.cuda()
    adjs_labels = [a.cuda() for a in adjs_labels]
    adjs = [a.cuda() for a in adjs]
    shared_feature = shared_feature.cuda()
    shared_feature_label = shared_feature_label.cuda()
    xs = [x.cuda() for x in xs]
    as_input = [a.cuda() for a in as_input]
device = adjs_labels[0].device

model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if train:
    # =========================================Train=============================================================
    bad_count = 0
    best_loss = 100
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0
    loss = 0.
    kl_step = 1.
    kl_max = 10000
    l = 0.0

    XXs = []
    AAs = []
    for v in range(num_graph):
        x = minmaxnormalize(rw_normalize(torch.mm(shared_feature_label, shared_feature_label.T)))
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        XXs.append(x)

        a = minmaxnormalize(rw_normalize(torch.mm(adjs_labels[v], adjs_labels[v].T)))
        AAs.append(a)

    weighh = [1e-12 for i in range(num_graph)]
    weights_h = normalize_weight(weighh)
    weights_S = []
    for v in range(num_graph):
        weightss = [1e-12, 1e-12]
        weightsS = normalize_weight(weightss)
        weights_S.append(weightsS)

    with torch.no_grad():
        model.eval()
        zx_norms, homo_xs, za_norms, homo_as, hs, h_all, qgs, x_preds, Ss = model(xs, adjs_labels, weights_S, weights_h)
        kmeans = KMeans(n_clusters=class_num, n_init=3)
        for v in range(num_graph):
            y_pred_g = kmeans.fit_predict(hs[v].data.cpu().numpy())
            model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
        y_pred_g = kmeans.fit_predict(hs[-1].data.cpu().numpy())
        model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch_num in range(epoch):
        model.train()
        loss = 0.
        loss_re_x = 0.
        loss_ho_x = 0.
        loss_ho_a = 0.
        loss_kl_g = 0.

        zx_norms, homo_xs, za_norms, homo_as, hs, h_all, qgs, x_preds, Ss = model(xs, adjs_labels, weights_S, weights_h)

        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_prim = kmeans.fit_predict(h_all.detach().cpu().numpy())
        pseudo_label = y_prim
        for v in range(num_graph):
            y_pred = kmeans.fit_predict(hs[v].detach().cpu().numpy())
            a = eva(y_prim, y_pred, visible=False, metrics='acc')
            weighh[v] = a
        weights_h = normalize_weight(weighh, p=weight_soft_h)

        sim_h = torch.mm(h_all, h_all.T)
        for v in range(num_graph):
            weightss[0] = F.cosine_similarity(homo_xs[v], sim_h).mean().detach().cpu().numpy()
            weightss[1] = F.cosine_similarity(homo_as[v], sim_h).mean().detach().cpu().numpy()
            weightsS = normalize_weight(weightss, p=weight_soft_S)
            weights_S[v] = weightsS

        for v in range(num_graph):
            loss_ho_x += F.mse_loss(homo_xs[v], XXs[v])
            loss_ho_a += F.mse_loss(homo_as[v], AAs[v])

        pgh = model.target_distribution(qgs[-1])
        loss_kl_g += F.kl_div(qgs[-1].log(), pgh, reduction='batchmean')
        for v in range(num_graph):
            loss_re_x += F.binary_cross_entropy(x_preds[v], xs[v])
            pg = model.target_distribution(qgs[v])
            loss_kl_g += F.kl_div(qgs[v].log(), pg, reduction='batchmean')
            loss_kl_g += F.kl_div(qgs[v].log(), pgh, reduction='batchmean')
        if l < kl_max:
            l = kl_step * epoch_num
        else:
            l = kl_max
        loss_kl_g *= l
        loss += 1 * loss_re_x + 1 * loss_kl_g + 1 * loss_ho_a + 1 * loss_ho_x

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        print(
            'epoch: {}, loss: {:.4f}, loss_re_x: {:.4f}, loss_ho_x:{:.4f}, loss_ho_a: {:.4f}, loss_kl_g: {:.4f}, badcount: {}'.format(
                epoch_num, loss, loss_re_x, loss_ho_x, loss_ho_a, loss_kl_g, bad_count
            ))

        if epoch_num % update_interval == 0:
            model.eval()
            zx_norms, homo_xs, za_norms, homo_as, hs, h_all, qgs, x_preds, Ss = model(xs, adjs_labels, weights_S, weights_h)
            kmeans = KMeans(n_clusters=class_num, n_init=10)
            y_eval = kmeans.fit_predict(h_all.detach().cpu().numpy())
            nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz', visible=False)

        if acc > best_acc:
            if os.path.exists('./pkl/SMHGC_{}_acc{:.4f}.pkl'.format(dataset, best_acc)):
                os.remove('./pkl/SMHGC_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            best_loss = loss
            bad_count = 0

            torch.save({'state_dict': model.state_dict(),
                        'weights_S': weights_S,
                        'weights_h': weights_h,
                        'datasets': dataset,
                        'lrs': lr,
                        'weight_decays': weight_decay,
                        'K': k_for_disc},
                       './pkl/SMHGC_{}_acc{:.4f}.pkl'.format(dataset, best_acc))

            print(
                'best acc:{:.4f}, best nmi:{:.4f}, best ari:{:.4f}, best f1:{:.4f}, best loss:{:.4f}, bestepoch:{}'.format(
                    best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
        else:
            bad_count += 1

        if bad_count >= patience:
            print(
                'complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
                    best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
            print()
            break

if not train:
    model_name = settings.model_name
else:
    model_name = 'SMHGC_{}_acc{:.4f}'.format(dataset, best_acc)

best_model = torch.load('./pkl/'+model_name+'.pkl', map_location=shared_feature.device)
state_dic = best_model['state_dict']
weights_S = best_model['weights_S']
weights_h = best_model['weights_h']

model.load_state_dict(state_dic)

model.eval()
with torch.no_grad():
    zx_norms, homo_xs, za_norms, homo_as, hs, h_all, qgs, x_preds, Ss = model(xs, adjs_labels, weights_S, weights_h)
    kmeans = KMeans(n_clusters=class_num, n_init=500)
    y_eval = kmeans.fit_predict(h_all.detach().cpu().numpy())
    nmi, acc, ari, f1 = eva(y, y_eval, 'Final Kz')

print('Test complete...')
