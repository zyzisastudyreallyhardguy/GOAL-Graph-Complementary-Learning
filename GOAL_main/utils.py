import random
import torch
import numpy as np


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def split_dataset(X, train_size=0.6, val_size=0.2, test_size=0.2):
    # Create a random permutation of the indices of the samples
    indices = np.random.permutation(X.shape[0])

    # Split the indices into the desired number of sets
    train_indices = indices[:int(train_size * X.shape[0])]
    val_indices = indices[int(train_size * X.shape[0]):int((train_size + val_size) * X.shape[0])]
    test_indices = indices[int((train_size + val_size) * X.shape[0]):]

    return train_indices, val_indices, test_indices

def knn_construct_graph(embed, k, idx_train=None, largest=True, sampling_ratio=0.1):
    n = embed.shape[0]

    if idx_train is not None:
        embed = embed[idx_train]

    # Randomly sample a subset of node indices
    num_samples = int(n * sampling_ratio)
    sampled_indices = random.sample(range(n), num_samples)

    # Calculate the dot product similarities for the sampled nodes
    similarities = torch.matmul(embed, embed[sampled_indices].T)

    if largest:
        topk_sim, topk_indices = torch.topk(similarities, k, dim=1)
    else:
        topk_sim, topk_indices = torch.topk(-similarities, k, dim=1)

    # Map the topk_indices back to the original node indices
    topk_lst = torch.tensor([[sampled_indices[i] for i in row] for row in topk_indices.tolist()], dtype=torch.long)

    return topk_lst


# def knn_construct_graph_train(embed, k, idx_train = None, largest = True):
#     train_indices = torch.where(data.train_mask == 1)[0]
#     train_embedding = embed[train_indices]
#     sim_mat = torch.matmul(embed.unsqueeze(0), train_embedding.t())  # get the sim_mat
#     topk = torch.topk(sim_mat, k, largest = largest).indices
#     topk_lst = train_indices[topk]

#     return topk_lst.squeeze(0)

#convert top neighbour list to edge_index graph
def convert_topk_2_graph(topk_lst, k):
    row_idx = torch.arange(0, topk_lst.shape[0]).repeat(k).unsqueeze(0)

    col_idx = topk_lst.T.reshape(topk_lst.shape[0]*k).unsqueeze(0)

    edge_index = torch.cat([row_idx, col_idx], 0)

    return edge_index

#convert top neighbour list (training set) to edge index graph
def convert_topk_2_graph_other(topk_lst, idx_train, k):
    row_idx = torch.arange(0, topk_lst.shape[0]).repeat(k).unsqueeze(0)
    col_idx = torch.LongTensor(idx_train[topk_lst].T.reshape(topk_lst.shape[0]*k)).unsqueeze(0)

    edge_index = torch.cat([row_idx, col_idx], 0)

    return edge_index

def check_homophily(edge_index, labels):
    homophily = (labels[edge_index[0]] == labels[edge_index[1]]).sum()
    homophily /= edge_index.shape[1]
    print('The homophily of the graph is:' + str(homophily))
