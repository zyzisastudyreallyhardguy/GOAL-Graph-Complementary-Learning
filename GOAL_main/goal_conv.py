import argparse

import torch
import torch.nn as nn

import os.path as osp
import numpy as np

import dgl
import copy
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset

import torch_geometric.transforms as T
import torch_geometric.utils as U
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor

from model import GOAL

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
parser.add_argument('--patience', type=int, default = 100)

args = parser.parse_args()
print(f"Training with DGL built-in GraphConv module.")

def evaluate(g, g_high, features, labels, mask, model, a, b, c, d):
    model.eval()
    with torch.no_grad():
        logits = model(g, g_high, features, a, b, c, d)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, g_high, features, labels, masks, model, epochs, a, b, c, d,learning_rate = 0.01, weight_decay = 5e-4):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # parameters for early stopping
    patience = args.patience # how many epochs to wait for improvement before stopping
    best_val_acc = 0  # track the best validation accuracy
    wait = 0  # track how many epochs without improvement
    
    model_path = 'data/' + args.dataset + '_model_' + args.gpu + '.pth'
    torch.save(model.state_dict(), model_path)

    # training loop
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


if __name__ == "__main__":
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

        # model training
        print("Training...")
        train(g0, g_high, features, labels, masks, model, args.epochs, args.alpha, args.beta, args.gamma, args.delta, args.lr, args.weight_decay)

        model.load_state_dict(torch.load('data/' + args.dataset + '_model_' + args.gpu + '.pth'))
        # test the model
        print("Testing...")
        acc = evaluate(g0, g_high, features, labels, masks[2], model, args.alpha, args.beta, args.gamma, args.delta)
        acc_results.append(acc)
        print("Test accuracy {:.4f}".format(acc))

    with open('result/gslog'  + args.ind +  '_{}.txt'.format(args.dataset), 'a') as f:
        f.write('\n' + str(np.mean(acc_results)))
        # f.write(str(np.std(acc_results)) + '\n')