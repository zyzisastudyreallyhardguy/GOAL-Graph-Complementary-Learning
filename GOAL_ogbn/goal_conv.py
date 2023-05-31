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

from ogb.nodeproppred import DglNodePropPredDataset

from model import GOAL

def evaluate(g, g_high, features, labels, mask, model, a, b, c, d):
    model.eval()
    with torch.no_grad():
        logits = model(g, g_high, features, a, b, c, d)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels.squeeze(1))
        return correct.item() * 1.0 / len(labels.squeeze(1))


def train(g, g_high, features, labels, masks, model, epochs, a, b, c, d, learning_rate = 0.01, weight_decay = 5e-4):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Early stopping parameters
    patience = 200
    best_val_acc = 0
    epochs_no_improve = 0

    model_path = 'data/' + args.dataset + '_model_' + args.gpu + '.pth'

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(g, g_high, features, a, b, c, d)
        if args.dataset not in ['ogbn-arxiv']:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        else:
            loss = loss_fcn(logits[train_mask], labels.squeeze(1)[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        acc = evaluate(g, g_high, features, labels, val_mask, model, a, b, c, d)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        
        # Early stopping and model saving
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break

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

    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")


    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0]

    train_idx = split_idx['train']
    train_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = split_idx['valid']
    val_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = split_idx['test']
    test_mask = torch.zeros((g.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_idx] = True

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    g.ndata['label'] = labels
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    g.ndata['feat'] = features.to(device)
    g.ndata['label'] = labels.to(device)

    features = features.to(device)
    labels = labels.to(device)

    if args.dataset not in ['ogbn-arxiv']:
        masks = torch.load('data/' + args.dataset + '_indices_' + args.gpu + '.pt')
    else:
        masks = [split_idx['train'], split_idx['valid'], split_idx['test']]

    g0 = copy.copy(g)
    if args.dataset not in ['ogbn-arxiv']:
        g0.remove_nodes([i for i in range(g0.ndata['feat'].shape[0])])
    if args.homo_gen:
        src_lst = torch.load('data/' + args.dataset + '_homo_'  + args.gpu + '.pt')[0].tolist()
        dst_lst = torch.load('data/' + args.dataset + '_homo_'  + args.gpu + '.pt')[1].tolist()
        g0.add_edges(src_lst, dst_lst)  # g0 homo graph
        g0.add_edges(dst_lst, src_lst)
    else:
        src_lst = g.edges()[0].tolist()
        dst_lst = g.edges()[1].tolist()
        g0.add_edges(src_lst, dst_lst)

    g0 = dgl.add_self_loop(g0)

    #high graph
    # g_high = data[0]
    g_high = copy.copy(g)
    g_high.remove_nodes([i for i in range(g_high.ndata['feat'].shape[0])])

    src_lst = torch.load('data/' + args.dataset + '_heter_' + args.gpu + '.pt')[0].tolist()
    dst_lst = torch.load('data/' + args.dataset + '_heter_' + args.gpu + '.pt')[1].tolist()
    g_high.add_edges(src_lst, dst_lst)
    g_high.add_edges(dst_lst, src_lst)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    g_high = g_high.int().to(device)

    acc_results = []
    for i in range(args.n_trials):
        # create GOAL model
        in_size = features.shape[1]
        out_size = g.ndata['label'].max().item() + 1
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