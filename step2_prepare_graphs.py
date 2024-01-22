#%%
import os
import pickle
import dill
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

#%%
def extend_labels(data_idx, n_win):
    ex_data_idx = []
    for idx in data_idx:
        st = idx * n_win
        ex_data_idx.extend(np.linspace(st, st + n_win - 1, n_win, dtype=int))
    return np.array(ex_data_idx)

def get_fold_indices(labels):
    fold_idx = {}
    skf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (train_idx_out, test_idx_out) in enumerate(skf_out.split(np.zeros(len(labels)), labels)):
        skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for j, (train_idx_in, val_idx_in) in enumerate(skf_in.split(np.zeros(len(labels[train_idx_out])), labels[train_idx_out])):
            fold_idx[f'outer{i+1}_inner{j+1}'] = {
                'train': train_idx_out[train_idx_in],
                'test': test_idx_out,
                'val': train_idx_out[val_idx_in]
            }
    return fold_idx

def load_data():
    with open('ldw_data/LDW_abide_data.pkl', 'rb') as f:
        data_dict = dill.load(f)
    return data_dict['node_feat'], data_dict['adj_mat'], data_dict['labels']

def get_fold_data(data, adj, seqlen, label, indices, name, i, j):
    idx = indices[f'outer{i}_inner{j}'][name]
    return [data[k] for k in idx], [adj[k] for k in idx], [seqlen[k] for k in idx], [label[k] for k in idx]

def pad_graph_seq(data):
    sequences = [torch.tensor(np.array(i)) for i in data]
    seqlengths = torch.LongTensor([len(x) for x in sequences])
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return sequences_padded, seqlengths

def convert2graphs(Corr, Adj, SeqLen, Label):
    n_sub = len(Corr)
    n_win = Corr[0].shape[0]
    num_classes = len(set(Label))
    Graph = np.empty([n_sub, n_win], dtype=object)

    for i in tqdm(range(n_sub)):
        for j in range(n_win):
            corr = Corr[i][j]
            adj = Adj[i][j]
            label = Label[i]
            padding = (torch.sum(adj, axis=1) == 0).all()

            nodeFeat = corr.float()
            index = torch.nonzero(adj)
            values = adj[index[:,0],index[:,1]]

            edgeIndex = index.T.long()
            edgeAttr = values.unsqueeze(-1).float()
            adj_node = adj.float()
            labeL = torch.tensor(label).long()

            graph = Data(x=nodeFeat, edge_index=edgeIndex, edge_attr=edgeAttr, adj=adj_node, y=labeL, 
                         num_nodes=nodeFeat.shape[0], num_node_features=nodeFeat.shape[1], num_classes=num_classes,
                         pad=padding, last=(j+1==SeqLen[i]))
            Graph[i, j] = graph

    return Graph

#%%
SEED = 42
np.random.seed(SEED)

saveTo = 'folds_data/'
os.makedirs(saveTo, exist_ok=True)

print('Loading data')
all_data, all_adj, labels = load_data()
all_data_padded, seqlengths = pad_graph_seq(all_data)
all_adj_padded, _ = pad_graph_seq(all_adj)
fold_indices = get_fold_indices(labels)

#%%
for i in range(1, 6):
    for j in range(1, 6):
        print(f'Processing outer{i}_inner{j}')
        train_data = get_fold_data(all_data_padded, all_adj_padded, seqlengths, labels, fold_indices, 'train', i, j)
        train_graphs = convert2graphs(*train_data)

        test_data = get_fold_data(all_data_padded, all_adj_padded, seqlengths, labels, fold_indices, 'test', i, j)
        test_graphs = convert2graphs(*test_data)

        val_data = get_fold
