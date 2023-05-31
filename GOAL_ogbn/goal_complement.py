import argparse
import copy
import os.path as osp

import torch
import torch.optim as optim
import random
import copy
import numpy as np
import random

import dgl
import dgl.function as fn

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric import utils as U
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomLinkSplit

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from metrics import NDCG
from model import GCN, GNN_high, ReconstructEncoder, listNet
from utils import knn_construct_graph, convert_topk_2_graph, check_homophily, index_to_mask, split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--use_graph', action='store_true')
parser.add_argument('--use_gnn_high', action='store_true')
parser.add_argument('--graph_gen', action='store_true')
parser.add_argument('--batch_test', action='store_true')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--k_homo', type=int, default=10)
parser.add_argument('--k_heter', type=int, default=10)
parser.add_argument('--batch_test_size', type=int, default=10000)
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--graph_gen_type', type=str, default='whole')
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--hidden_gnn', type=int, default=16)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs_gnn', type=int, default=200)
parser.add_argument('--epochs_gen', type=int, default=200)
parser.add_argument('--num_layers_gnn', type=int, default=3)
parser.add_argument('--num_layers_gen', type=int, default=3)
parser.add_argument('--train_sample_ratio', type=float, default=0.1)
parser.add_argument('--edge_sample_ratio', type=float, default=0.1)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    transform = T.Compose([T.NormalizeFeatures()])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform = transform)
    data = dataset[0]
    if args.dataset == 'pubmed':
        # add self-loop
        data.edge_index = U.add_self_loops(data.edge_index)[0]
elif args.dataset in ['ogbn-arxiv', 'ogbn-products']:
    dataset = PygNodePropPredDataset(name = args.dataset)
    data = dataset[0]
    data.edge_index = U.to_undirected(data.edge_index)
    data.edge_index = U.add_self_loops(data.edge_index)[0]
    data.x = data.x.to(torch.float)
    data.y = data.y.squeeze()
    split_idx = dataset.get_idx_split()
    data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
    data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)
    data = data.to(device)
    evaluator = Evaluator(args.dataset)

labels = data.y.cpu().numpy()

if args.dataset not in ['ogbn-arxiv', 'ogbn-products']:
    indices = split_dataset(data.x)
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        data.train_mask[:] = False
        data.train_mask[indices[0]] = True
        data.val_mask[:] = False
        data.val_mask[indices[1]] = True
        data.test_mask[:] = False
        data.test_mask[indices[2]] = True

    torch.save(indices, 'data/' + args.dataset + '_indices_' + args.gpu + '.pt')

if args.use_gnn_high:
    g = dgl.DGLGraph(data.edge_index.T.tolist())
    g.ndata['feat'] = data.x
    g.ndata['label'] = data.y
    g = g.to(torch.device('cuda:' + args.gpu))
    model = GNN_high(data.x.shape[1], args.hidden, args.hidden, args.num_layers_gnn).to(device)
else:
    model = GCN(data.x.shape[1], args.hidden_gnn, labels.max() + 1, args.num_layers_gnn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    if args.use_gnn_high:
        out, embed = model(data.x, g, gamma = args.gamma)
    else:
        out, embed = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    if args.use_gnn_high:
        pred, embed = model(data.x, g, gamma = args.gamma)
    else:
        pred, embed = model(data.x, data.edge_index)
    pred = pred.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs, embed

if args.pretrain:
    data = data.cuda()
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs_gnn + 1):
        loss = train()
        [train_acc, val_acc, tmp_test_acc], embed = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            torch.save(embed, 'data/' + args.dataset + '_embed_' + args.gpu + '.pt')
    data = data.cpu()

'''preprocessing obtain ranking list for every node(sample)'''
'''the ranking list include the top 5 most similar same class nodes and different class nodes'''
sample_fraction = args.train_sample_ratio
train_idx = data.train_mask.nonzero().T.squeeze(0) #train_idices
sample_size = int(len(train_idx) * sample_fraction)
sampled_train_idx = random.sample(train_idx.tolist(), sample_size)

#create a mask for the sampled training set
sampled_train_mask = torch.zeros_like(data.train_mask, dtype = torch.bool)
sampled_train_mask[sampled_train_idx] = True

#compute the similarity matrix using the sampled training set
sim_mat = data.x @ data.x[sampled_train_mask].T #every node and training set sim
data_train_y = data.y[sampled_train_mask] #get training set labels

#preprocessing to obtain the ranking list for every node
rank_lst = [[] for i in range(data.x.shape[0])]

#get the lowest number of training samples in the same class (max version)
if args.dataset == 'pubmed':
    list_len = 200
else:
    list_len = int(torch.FloatTensor([(data_train_y == i).sum() for i in range(data.y.max() + 1)]).min().item())

sampled_train_idx = torch.LongTensor(sampled_train_idx)

for i in range(data.x.shape[0]):
    #find the list_len most similar same class nodes
    if i in sampled_train_idx:
        sim_mat[i][(sampled_train_idx == i).nonzero().item()] = 0
    pos_lst = torch.topk(sim_mat[i][data_train_y == data.y[i]],list_len).indices #find most similar nodes indices in the sampled training set
    rank_lst[i] += (data_train_y == data.y[i]).nonzero().T.squeeze(0)[pos_lst].tolist() #get add indices

    #find the list_len most similar different class nodes
    neg_lst = torch.topk(sim_mat[i][data_train_y != data.y[i]],list_len).indices #find most similar dis-class nodes in the sampled training set
    rank_lst[i] += (data_train_y != data.y[i]).nonzero().T.squeeze(0)[neg_lst].tolist() #get add indices

'''load GNN trained embed and use as data.x'''
if args.pretrain:
    gnn_embed = torch.load('data/' + args.dataset + '_embed_' + args.gpu + '.pt').cpu()
    data.x = gnn_embed

print('Pretraining Finished!')

# create training edge set and test edge set
'''training/test(scalable ver)'''

# Function to create edge list based on same labels with random sampling
def create_edges(data, node_mask, sampling_ratio=0.1):
    edge_list = []

    masked_node_indices = node_mask.nonzero().T.squeeze(0)
    node_labels = data.y[node_mask].tolist()
    label_to_indices = {}

    for idx, label in zip(masked_node_indices, node_labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    for i, label in zip(masked_node_indices, node_labels):
        same_label_indices = label_to_indices[label]
        num_sampled_edges = int(len(same_label_indices) * sampling_ratio)
        sampled_indices = random.sample(same_label_indices, num_sampled_edges)
        if i in sampled_indices:
            sampled_indices.remove(i)
        edge_list.extend([(i, j) for j in sampled_indices])

    return torch.tensor(edge_list, dtype=torch.long).T

# Sampling ratio for the edges
sampling_ratio = args.edge_sample_ratio

# Create train_dataset using only training nodes
train_dataset = copy.deepcopy(data)
train_dataset.edge_index = create_edges(data, data.train_mask, sampling_ratio)

# Create the transform for the train_dataset
transform = RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, split_labels=True, add_negative_train_samples=False)
train_data, _, _ = transform(train_dataset)
del train_dataset

# Create test_dataset using only test nodes
test_dataset = copy.deepcopy(data)
test_dataset.edge_index = create_edges(data, ~data.train_mask, sampling_ratio)

# Create the transform for the test_dataset
transform = RandomLinkSplit(num_val=0.0, num_test=0.9999, is_undirected=True, split_labels=True, add_negative_train_samples=False)
_, val_data, test_data = transform(test_dataset)
del test_dataset

print('Data Splitting Finished!')

# test_data.edge_index = train_data.edge_index

'''use no graph'''
if not args.use_graph:
    def create_self_edges(num_nodes):
        edge_indices = torch.zeros((2, num_nodes), dtype=torch.long)
        for i in range(num_nodes):
            edge_indices[0, i] = i
            edge_indices[1, i] = i
        return edge_indices

    num_nodes = data.x.shape[0]
    train_data.edge_index = create_self_edges(num_nodes)
    test_data.edge_index = create_self_edges(num_nodes)

train_data = train_data.cuda()
test_data = test_data.cuda()

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, 2 * out_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(2 * out_channels, 2 * out_channels))
        self.convs.append(GATConv(2 * out_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if i != args.num_layers_gen - 1:
                x = conv(x, edge_index).relu()
            else:
                x = conv(x, edge_index)
        return x

in_channels, out_channels = data.x.shape[1], args.hidden

model = ReconstructEncoder(Encoder(in_channels, out_channels, args.num_layers_gen))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
CE_loss = nn.CrossEntropyLoss()

ndcg_gain_in_train="exp2"
ideal_dcg = NDCG(2**9, ndcg_gain_in_train)
sigma = 1.0
N = 1.0 / ideal_dcg.maxDCG([i + 1 for i in range(10)])

def compute_loss_ndcg(z, data, rank_lst, batch_test = False):
    if not batch_test:
        y_pred = (z.unsqueeze(1)[data.train_mask] @ z[rank_lst][data.train_mask].permute(0,2,1)).squeeze(1) #y_pred_batch
        #training set embedding @ training set all most similar same/different class embed
    else:
        sample_nodes = 2000
        train_indices = data.train_mask.nonzero().squeeze(1).tolist()
        random_nodes = random.sample(train_indices, sample_nodes)
        y_pred = (z.unsqueeze(1)[random_nodes] @ z[rank_lst][random_nodes].permute(0, 2, 1)).squeeze(1)
    y_true = []
    count = 0
    for i in range(4, 1, -1):
        count += int(0.2*list_len*2)
        # if i != 2:
        y_true += [i] * int(0.2*list_len*2)
        # else:
        #     y_true += [1] * int(0.2 * list_len * 2)

    y_true += [1]*(list_len * 2 - count)
    if not batch_test:
        y_true = torch.FloatTensor(y_true).repeat(data.train_mask.sum(), 1)
    else:
        y_true = torch.FloatTensor(y_true).repeat(sample_nodes, 1)
    loss_ndcg = listNet(y_pred, y_true.cuda())
    return loss_ndcg

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    if not args.batch_test:
        loss_recon = model.recon_loss(z, train_data.pos_edge_label_index)

        '''calculate lambda loss'''
        loss_ndcg = compute_loss_ndcg(z, data, rank_lst)
        # print(loss_ndcg)
    else:
        numlist = random.sample(np.arange(train_data.pos_edge_label_index.shape[1]).tolist(), 10000) #randomly find 10000 edges

        #get batched edges source and dist list
        batch_pos_src = train_data.pos_edge_label_index[0][numlist]
        batch_pos_dst = train_data.pos_edge_label_index[1][numlist]

        batch_pos = torch.cat((batch_pos_src.unsqueeze(0), batch_pos_dst.unsqueeze(0)), dim=0)

        loss_recon = model.recon_loss(z, batch_pos)
        loss_ndcg = compute_loss_ndcg(z, data, rank_lst, args.batch_test)

    loss_ce = CE_loss(z[data.train_mask], data.y[data.train_mask].cuda())

    loss = loss_ndcg + loss_recon + loss_ce

    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data, batch_test):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    if not batch_test:
        print('top 1 accuracy result')
        pred = data.y[(z[data.train_mask] @ z[data.train_mask].T).argmax(0)]
        pred_ratio = ((data.y[data.train_mask] == pred).sum())/data.train_mask.shape[0]
        print(pred_ratio)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index), z
    else:
        len_batch = (data.neg_edge_label_index.shape[1])//args.batch_test_size
        fast_test = random.sample(np.arange(len_batch - 1).tolist(), 10)
        auc_list = []
        prc_list = []
        # for batch in range(len_batch):
        for batch in fast_test:
            # if batch <= len_batch - 2:
            batch_pos_src = data.pos_edge_label_index[0][batch*args.batch_test_size:(batch + 1)*args.batch_test_size]
            batch_pos_dst = data.pos_edge_label_index[1][batch*args.batch_test_size:(batch + 1)*args.batch_test_size]

            batch_pos = torch.cat((batch_pos_src.unsqueeze(0), batch_pos_dst.unsqueeze(0)), dim=0)

            batch_neg_src = data.neg_edge_label_index[0][batch*args.batch_test_size:(batch + 1)*args.batch_test_size]
            batch_neg_dst = data.neg_edge_label_index[1][batch*args.batch_test_size:(batch + 1)*args.batch_test_size]

            batch_neg = torch.cat((batch_neg_src.unsqueeze(0), batch_neg_dst.unsqueeze(0)), dim=0)

            auc_temp, prc_temp = model.test(z, batch_pos, batch_neg)

            auc_list.append(auc_temp)
            prc_list.append(prc_temp)

        auc = np.mean(auc_list)
        prc = np.mean(prc_list)

        return (auc, prc), z

print(args)
for epoch in range(1, args.epochs_gen + 1):
    loss = train()
    (auc, ap), z = test(test_data, args.batch_test)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    if epoch == (args.epochs_gen - 1):
        sig = nn.Sigmoid()
        torch.save(z, 'data/' + args.dataset + '_embed_ae_'  + args.gpu + '.pt')

#use learned embedding to evaluate performance
embed = torch.load('data/' + args.dataset + '_embed_ae_'  + args.gpu + '.pt').cpu()

if args.graph_gen:
    #get edge index for homo graph
    topk_lst = knn_construct_graph(embed, args.k_homo, largest=True)
    gen_homo_graph = convert_topk_2_graph(topk_lst, args.k_homo)

    # get edge index for heter graph
    topk_lst = knn_construct_graph(embed, args.k_heter, largest=False)
    gen_heter_graph = convert_topk_2_graph(topk_lst, args.k_heter)

    print('Graph Generation Finished!')

    #whole generated graph homophily rate
    check_homophily(gen_homo_graph, labels)
    check_homophily(gen_heter_graph, labels)

    #test set generated graph homophily rate
    def obtain_test_edges(edge_index):
        source_nodes = edge_index[0]
        test_edges = data.test_mask[source_nodes]
        return edge_index[:, test_edges]

    check_homophily(obtain_test_edges(gen_homo_graph), labels)
    check_homophily(obtain_test_edges(gen_heter_graph), labels)

    torch.save(gen_homo_graph, 'data/' + args.dataset + '_homo_' + args.gpu + '.pt')
    torch.save(gen_heter_graph, 'data/' + args.dataset + '_heter_' + args.gpu + '.pt')