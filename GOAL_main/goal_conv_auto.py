import argparse

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import os.path as osp
import numpy as np

import dgl
import copy
import dgl.nn as dglnn
import dgl.function as fn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor

from skopt import Optimizer
from skopt.utils import use_named_args
from skopt.space import Real

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

class GOAL(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layer):
        super().__init__()
        self.batchnorm = nn.ModuleList()

        self.mp = MPConv()
        self.layers = nn.ModuleList()
        # two-layer GOAL
        self.layers.append(nn.Linear(in_size, hid_size, bias=False))
        self.batchnorm.append(nn.BatchNorm1d(hid_size))

        for i in range(num_layer - 2):
            self.layers.append(nn.Linear(hid_size, hid_size, bias=False))
            self.batchnorm.append(nn.BatchNorm1d(hid_size))
        self.layers.append(nn.Linear(hid_size, out_size, bias=False))
        self.dropout = nn.Dropout(0.5)

        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.gamma = nn.Parameter(torch.Tensor(1))
        self.delta = nn.Parameter(torch.Tensor(1))
        self.batchnorm.append(nn.BatchNorm1d(out_size))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.alpha)
        torch.nn.init.uniform_(self.beta)
        torch.nn.init.uniform_(self.gamma)
        torch.nn.init.uniform_(self.delta)

    def regularization_loss(self, reg_coeff):
        reg_loss = reg_coeff * (self.alpha ** 2 + self.beta ** 2 + self.gamma ** 2 + self.delta ** 2)
        return reg_loss

    def forward(self, g, g_high, features, a, b, c, d):
        h = features

        h = F.normalize(h)

        for i, layer in enumerate(self.layers):
            temp_h = copy.copy(h)
            h_p = self.mp(g, h)
            h_n = F.normalize(self.mp(g_high, h))
            h_p = self.layers[i](h_p)
            h_n =  - self.layers[i](h_n)
            h_i = self.layers[i](temp_h)
            h = h_p + self.beta*F.normalize(h_n) + self.gamma*h_i - self.delta*F.normalize(self.mp(g_high, h_p))
            h = self.batchnorm[i](h)

        return h

def evaluate(g, g_high, features, labels, mask, model, a, b, c, d):
    model.eval()
    with torch.no_grad():
        logits = model(g, g_high, features, a, b, c, d)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, g_high, features, labels, masks, model, epochs, a, b, c, d, learning_rate = 0.01, weight_decay = 5e-4):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training loop
    wait = 0
    patience = 500
    best_val_acc = 0
    model_path = 'data/' + args.dataset + '_model_' + args.gpu + '.pth'
    torch.save(model.state_dict(), model_path)

    for epoch in range(epochs):
        model.train()
        logits = model(g, g_high, features, a, b, c, d)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # validation
        val_acc = evaluate(g, g_high, features, labels, val_mask, model, a, b, c, d)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), val_acc
            )
        )
        
        # early stopping check
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            wait = 0
            #save the model
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait == patience:
                print(f"Early stopping at epoch {epoch}")
                break

def train_param(g, g_high, features, labels, masks, model, epochs, a, b, c, d, learning_rate = 0.01, weight_decay = 5e-4):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training loop
    model.train()
    logits = model(g, g_high, features, a, b, c, d)
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = evaluate(g, g_high, features, labels, val_mask, model, a, b, c, d)
    print(
        "Loss {:.4f} | Accuracy {:.4f} ".format(
            loss.item(), acc
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument('--homo_gen', action='store_true', help="use generated homo graph")
    parser.add_argument('--hid_units', type=int, default=512, help='hidden dimension')
    parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type = float, default=5e-4, help='weight decay')
    parser.add_argument('--alpha', type=float, default=1, help='control A*')
    parser.add_argument('--beta', type=float, default=1, help='control A_homo')
    parser.add_argument('--gamma', type=float, default=1, help='control A_heter')
    parser.add_argument('--delta', type=float, default=1, help='control A_homo@A_heter')
    parser.add_argument('--n_trials', type=int, default=1, help='number of trails')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--n_layers', type=int, default=2, help='hidden dimension')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ind', type=str, default='0')
    parser.add_argument('--calls', type = int, default=20, help='number of calls')
    parser.add_argument('--ski_opt', action='store_true', help="use skit-opt")

    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    elif args.dataset == 'photo':
        data = AmazonCoBuyPhotoDataset(transform = transform)
    elif args.dataset == 'computers':
        data = AmazonCoBuyComputerDataset(transform = transform)
    elif args.dataset in ['chameleon', 'squirrel', 'actor']:
        pass
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        tg_data = dataset[0]

        g = data[0]

        g.ndata['feat'] = tg_data.x
        g.ndata["label"] = tg_data.y
        g.ndata["train_mask"] = tg_data.train_mask
        g.ndata["val_mask"] = tg_data.val_mask
        g.ndata["test_mask"] = tg_data.test_mask

        masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    elif args.dataset in ['computers', 'photo']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        tg_data = dataset[0]
        g = data[0]

        g.ndata['feat'] = tg_data.x
        g.ndata["label"] = tg_data.y
    elif args.dataset in ['chameleon', 'squirrel', 'actor']:
        if args.dataset in ['chameleon', 'squirrel']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WikipediaNetwork')
            dataset = WikipediaNetwork(path, args.dataset, transform=T.NormalizeFeatures())
        elif args.dataset in ['actor']:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Actor')
            dataset = Actor(path, transform=T.NormalizeFeatures())

        tg_data = dataset[0]
        tg_data.edge_index = U.to_undirected(tg_data.edge_index)
        g = dgl.DGLGraph(tg_data.edge_index.T.tolist())
        data = g

        g.ndata['feat'] = tg_data.x
        g.ndata["label"] = tg_data.y

        data.num_classes = tg_data.y.max().item() + 1

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]

    per_graph = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float()
    masks = torch.load('data/' + args.dataset + '_indices_' + args.gpu + '.pt')

    g0 = copy.copy(g)
    g0.remove_nodes([i for i in range(tg_data.x.shape[0])])
    if args.homo_gen:
        src_lst = torch.load('data/' + args.dataset + '_homo_'  + args.gpu + '.pt')[0].tolist()
        dst_lst = torch.load('data/' + args.dataset + '_homo_'  + args.gpu + '.pt')[1].tolist()
        g0.add_edges(src_lst, dst_lst)  # g0 homo graph
        g0.add_edges(dst_lst, src_lst)
    else:
        src_lst = tg_data.edge_index[0].tolist()
        dst_lst = tg_data.edge_index[1].tolist()
        g0.add_edges(src_lst, dst_lst)

    g0 = dgl.add_self_loop(g0)

    g_high = copy.copy(g)
    g_high.remove_nodes([i for i in range(g_high.ndata['feat'].shape[0])])

    if args.dataset not in ['chameleon', 'squirrel']:
        src_lst = torch.load('data/' + args.dataset + '_heter_' + args.gpu + '.pt')[0].tolist()
        dst_lst = torch.load('data/' + args.dataset + '_heter_' + args.gpu + '.pt')[1].tolist()
        g_high.add_edges(src_lst, dst_lst)
        g_high.add_edges(dst_lst, src_lst)
    else:
        src_lst = tg_data.edge_index[0].tolist()
        dst_lst = tg_data.edge_index[1].tolist()
        g_high.add_edges(src_lst, dst_lst)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    g_high = g_high.int().to(device)

    acc_results = []
    for i in range(args.n_trials):
        # create GOAL model
        in_size = features.shape[1]
        out_size = data.num_classes
        model = GOAL(in_size, args.hid_units, out_size, args.n_layers).to(device)

        if args.ski_opt:
            # Define the search space for alpha, beta, gamma, and delta
            space = [
                Real(0, 1, name="alpha"),
                Real(0, 1, name="beta"),
                Real(0, 1, name="gamma"),
                Real(0, 1, name="delta"),
            ]


            @use_named_args(space)
            def objective(**params):
                # Set the model parameters
                model.alpha.data.fill_(params["alpha"])
                model.beta.data.fill_(params["beta"])
                model.gamma.data.fill_(params["gamma"])
                model.delta.data.fill_(params["delta"])

                # Train and evaluate the model
                train_param(g0, g_high, features, labels, masks, model, args.epochs, args.alpha, args.beta, args.gamma, args.delta, args.lr, args.weight_decay)
                valid_acc = evaluate(g0, g_high, features, labels, masks[1], model, args.alpha, args.beta, args.gamma, args.delta)

                # Return the negative validation accuracy for minimization
                return -valid_acc


            # Initialize the optimizer with the search space
            opt = Optimizer(dimensions=space, base_estimator="GP")

            # Perform Bayesian Optimization
            n_calls = 30
            for _ in range(args.calls):
                suggested_params = opt.ask()
                loss = objective(suggested_params)
                opt.tell(suggested_params, loss)

            # Find the best parameters and set them in the model
            best_params = opt.Xi[np.argmin(opt.yi)]
            model.alpha.data.fill_(best_params[0])
            model.beta.data.fill_(best_params[1])
            model.gamma.data.fill_(best_params[2])
            model.delta.data.fill_(best_params[3])

        # model training
        print("Training...")
        train(g0, g_high, features, labels, masks, model, args.epochs, args.alpha, args.beta, args.gamma, args.delta, args.lr, args.weight_decay)

        # test the model
        print("Testing...")
        model_path = 'data/' + args.dataset + '_model_' + args.gpu + '.pth'
        model.load_state_dict(torch.load(model_path))
        acc = evaluate(g0, g_high, features, labels, masks[2], model, args.alpha, args.beta, args.gamma, args.delta)
        acc_results.append(acc)
        print("Test accuracy {:.4f}".format(acc))

    with open('result/gslog'  + args.ind +  '_{}.txt'.format(args.dataset), 'a') as f:
        f.write('\n' + str(np.mean(acc_results)))
        # f.write(str(np.std(acc_results)) + '\n')