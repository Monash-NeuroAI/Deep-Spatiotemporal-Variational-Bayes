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
def extend_labels(data_idx,n_win):
	ex_data_idx = []
	for i in range(len(data_idx)):
		idx = data_idx[i]
		st = idx * n_win
		en = st + n_win
		ex_data_idx.extend(np.linspace(st,en-1,n_win,dtype=int))
	return np.array(ex_data_idx)

def get_fold_indices(labels):
	fold_idx = dict()
	labels_out = labels

	skf_out = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
	for i, (train_idx_out, test_idx_out) in enumerate(skf_out.split(np.zeros(len(labels_out)), labels_out)):

		labels_in = labels_out[train_idx_out]
		
		skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
		for j, (train_idx_in, val_idx_in) in enumerate(skf_in.split(np.zeros(len(labels_in)), labels_in)):

			train_idx = train_idx_out[train_idx_in]
			val_idx = train_idx_out[val_idx_in]
			test_idx = test_idx_out

			fold_idx['outer{}_inner{}'.format(i+1,j+1)] = dict()
			fold_idx['outer{}_inner{}'.format(i+1,j+1)]['train'] = train_idx
			fold_idx['outer{}_inner{}'.format(i+1,j+1)]['test'] = test_idx
			fold_idx['outer{}_inner{}'.format(i+1,j+1)]['val'] = val_idx
		   
	return fold_idx

def load_data():
	with open('./ldw_data/LDW_abide_data.pkl', 'rb') as f:
		f = dill.load(f)
	data = f['node_feat']
	adj = f['adj_mat']
	labels = f['labels']
	return data, adj, labels

def get_fold_data(data, adj, seqlen, label, indices, name, i, j):
	data = [ data[i] for i in indices['outer{}_inner{}'.format(i,j)][name] ]
	adj = [ adj[i] for i in indices['outer{}_inner{}'.format(i,j)][name] ]
	seqlen = [ seqlen[i] for i in indices['outer{}_inner{}'.format(i,j)][name] ]
	label = [ label[i] for i in indices['outer{}_inner{}'.format(i,j)][name] ]
	return data, adj, seqlen, label

def pad_graph_seq(data):
	# Convert data to torch tensor
	sequences = [torch.tensor(np.array(i)) for i in data]
	# Sequence lengths T of batch (B,)
	seqlengths = torch.LongTensor([len(x) for x in sequences])
	# Pad sequences to same length (B[TxD] -> TxBxD) 
	sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
	# labels = torch.LongTensor(labels)
	return sequences_padded, seqlengths


def convert2graphs(Corr, Adj, SeqLen, Label):
	"""
	inputs
	  data: a list of node samples of node features
	outputs
	  cand_graphs: a list of graphs 
	"""
	n_sub = len(Corr)
	n_win = Corr[0].shape[0]
	num_classes = len(set(Label))
	
	# Initiate empty container
	Graph = np.empty([n_sub,n_win],dtype=object)

	for i in tqdm(range(n_sub)):
		for j in range(n_win):

			corr = Corr[i][j]
			adj = Adj[i][j]
			seqlen = SeqLen[i]
			label = Label[i]
			
			# Check if there are 0-in-degree nodes in the graph
			if (torch.sum(adj, axis=1) == 0).all(): padding = True
			else: padding = False

			# get the non-zero elements of adj matrix
			nodeFeat = corr.float()
			num_nodes = nodeFeat.shape[0]
			num_node_features = nodeFeat.shape[1]
			index = torch.nonzero(adj)
			values = adj[index[:,0],index[:,1]]

			# add edges to the graph
			edgeIndex = index.T.long()
			edgeAttr = values.unsqueeze(-1).float()
			adj_node = adj.float()
			labeL = torch.tensor(label).long()

			# Build graph
			graph = Data(x=nodeFeat, edge_index=edgeIndex, edge_attr=edgeAttr, adj=adj_node, y=labeL, 
			num_nodes=num_nodes, num_node_features=num_node_features, num_classes=num_classes,
			pad=padding, last=(j+1==seqlen))

			Graph[i,j] = graph

	return Graph

#%%
SEED = 42
np.random.seed(SEED)

saveTo = './folds_data/'  
os.makedirs(saveTo, exist_ok=True)

print('Loading data')
all_data, all_adj, labels = load_data()
all_data_padded, seqlengths = pad_graph_seq(all_data)
all_adj_padded, _ = pad_graph_seq(all_adj)
fold_indices = get_fold_indices(labels)

#%%
for i in range(1,6):
	for j in range(1,6):

		print('Processing outer{}_inner{}'.format(i,j))
		Corr, Adj, SeqLen, Label = get_fold_data(all_data_padded, all_adj_padded, seqlengths, labels, fold_indices, 'train', i, j)
		train_graphs = convert2graphs(Corr, Adj, SeqLen, Label)

		Corr, Adj, SeqLen, Label = get_fold_data(all_data_padded, all_adj_padded, seqlengths, labels, fold_indices, 'test', i, j)
		test_graphs = convert2graphs(Corr, Adj, SeqLen, Label)

		Corr, Adj, SeqLen, Label = get_fold_data(all_data_padded, all_adj_padded, seqlengths, labels, fold_indices, 'val', i, j)
		val_graphs = convert2graphs(Corr, Adj, SeqLen, Label)

		graphs = {}
		graphs['train_graphs'] = train_graphs
		graphs['test_graphs'] = test_graphs
		graphs['val_graphs'] = val_graphs

		print('Saving graphs ...')     
		with open(saveTo+'graphs_outer'+str(i)+'_inner'+str(j)+'.pkl', 'wb') as f:
			torch.save(graphs, f)
			f.close()
