#%%
import os
import time
import dill
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import det, svd, cholesky, cholesky_ex, inv, matrix_rank
from torch_geometric.nn import Sequential, GCNConv, TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import negative_sampling, batched_negative_sampling, to_dense_batch, unbatch 

def multiply():
	return 5 * 2

def get_device():
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.backends.cudnn.enabled = True
		return torch.device('cuda')
	else:
		return torch.device('cpu')

def get_activation(type='relu'):
	"""
	Return tensorflow activation function given string name.
	Args:
		type:
	Returns:
	"""
	if type == 'relu':
		return nn.ReLU()
	elif type == 'elu':
		return nn.ELU()
	elif type == 'lelu':
		return nn.LeakyReLU()
	elif type == 'tanh':
		return nn.Tanh()
	elif type == 'sigmoid':
		return nn.Sigmoid()
	elif type == 'softplus':
		return nn.Softplus()
	elif type == None:
		return None
	else:
		raise Exception("Activation function not supported.")

def dense_vary(input_dim, layer_dims, output_dim, activation='relu', dropout=0., batch_norm=False, last_act=None):
	layers = nn.Sequential()
	if len(layer_dims) != 0:
		layers.append(nn.Linear(input_dim, layer_dims[0]))
		layers.append(get_activation(activation))
		if batch_norm: layers.append(nn.BatchNorm1d(layer_dims[0]))
		layers.append(nn.Dropout(p=dropout))
		for i in range(len(layer_dims) - 1):
			layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
			layers.append(get_activation(activation))
			if batch_norm: layers.append(nn.BatchNorm1d(layer_dims[i+1]))
			layers.append(nn.Dropout(p=dropout))
		layers.append(nn.Linear(layer_dims[len(layer_dims) - 1], output_dim))
		if last_act is not None:
			layers.append(get_activation(last_act))
	else: 
		layers.append(nn.Linear(input_dim, output_dim))
		if last_act is not None:
			layers.append(get_activation(last_act))
	return layers

def GCN_vary(input_dim, layer_dims, output_dim, activation='relu', last_act=None):
	modules = []
	if len(layer_dims) != 0:
		modules.append((TransformerConv(input_dim, layer_dims[0]), 'x, edge_index -> x'))
		modules.append((get_activation(activation)))
		for i in range(len(layer_dims) - 1):
			modules.append((TransformerConv(layer_dims[i], layer_dims[i+1]), 'x, edge_index -> x'))
			modules.append((get_activation(activation)))
		modules.append((TransformerConv(layer_dims[len(layer_dims) - 1], output_dim), 'x, edge_index -> x'))
		if last_act is not None:
			modules.append(get_activation(last_act))
	else: 
		modules.append((TransformerConv(input_dim, output_dim), 'x, edge_index -> x'))
		if last_act is not None:
			modules.append(get_activation(last_act))
	layers = Sequential('x, edge_index', modules)
	return layers

def recurrent_cell(input_dim, hidden_dim, rnn_type):
	if rnn_type == 'lstm':
		return nn.LSTMCell(input_dim, hidden_dim)
	elif rnn_type == 'gru':
		return nn.GRUCell(input_dim, hidden_dim)
	else:
		raise Exception("No such rnn type.")

class GraphDecoder(nn.Module):
	def __init__(self, activation=None, dropout=0.5):
		super().__init__()
		self.act = get_activation(activation)
		self.dropout = dropout

	def forward(self, z, edge_index):
		z = F.dropout(z, self.dropout, training=self.training)
		value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
		if self.act is not None:
			value = self.act(value)
		return value
	
	def forward_sb(self, z):
		z = F.dropout(z, self.dropout, training=self.training)
		adj = z @ torch.transpose(z, dim0=-2, dim1=-1)
		if self.act is not None:
			adj = self.act(adj)
		return adj

	def cov2corr(self, cov):
		v = torch.sqrt(torch.diagonal(cov,dim1=-2, dim2=-1))
		outer_v = v.unsqueeze(dim=-1) @ v.unsqueeze(dim=-2)
		corr = cov / outer_v
		return corr

	def loss(self, input, target):
		temp_sum = target.sum()
		temp_size = target.shape[0]
		weight = float(temp_size * temp_size - temp_sum) / temp_sum
		norm = (temp_size * temp_size) / float((temp_size * temp_size - temp_sum) * 2)
		nll_loss_mat = F.binary_cross_entropy_with_logits(input=input, target=target, pos_weight=weight, reduction='none')
		nll_loss = norm * torch.mean(nll_loss_mat)
		return nll_loss

	def loss_sb(self, input, target, reduce=False):
		temp_sum = input.sum(dim=[-1,-2])
		temp_size = input.shape[1]
		weight = (temp_size * temp_size - temp_sum) / temp_sum
		norm = (temp_size * temp_size) / ((temp_size * temp_size - temp_sum) * 2)
		nll_loss_mat = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
		nll_loss = norm * weight * torch.mean(nll_loss_mat, dim=[-1,-2])
		if reduce: nll_loss = torch.mean(nll_loss)
		return nll_loss

class GraphClassifier(nn.Module):
	def __init__(self, input_dim, layer_dims, output_dim, dropout=0., batch_norm=False):
		super().__init__()
		self.dropout = dropout
		self.linear = dense_vary(input_dim, layer_dims, output_dim, last_act=None, dropout=dropout, batch_norm=batch_norm)

	def forward(self, input, target, sample=False):
		y_logit = self.linear(input)
		y_bce = self.loss(y_logit, target)
		with torch.no_grad():
			y_acc, y_prob, y_pred = self.acc(y_logit, target)
		y_BCE = y_bce.mean(); y_ACC = y_acc.mean()
		if sample:
			return y_BCE, y_ACC, y_prob, y_pred
		else:
			return y_BCE, y_ACC

	def loss(self, logit, target, reduce=False):
		if logit.shape[-1] > 1:
			nll_loss = F.cross_entropy(input=logit, target=target, reduction='none')
		else:
			nll_loss = F.binary_cross_entropy_with_logits(input=logit.squeeze(-1), target=target.float(), reduction='none')
		if reduce: nll_loss = torch.mean(nll_loss)
		return nll_loss

	def acc(self, logit, target, reduce=False):
		if logit.shape[-1] > 1:
			prob = F.softmax(logit, dim=-1)
			pred = torch.argmax(prob, dim=-1).float()
			acc = torch.abs(target - pred)
		else:
			prob = torch.sigmoid(logit.squeeze(-1))
			pred = torch.round(prob)
			acc = torch.abs(target - pred)
		if reduce: acc = torch.mean(acc); prob = torch.mean(prob)
		return acc, prob, pred

class GRU_GCN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layer=1, bias=True):
		super().__init__()

		self.hidden_size = hidden_size
		self.n_layer = n_layer
		
		# gru weights
		self.weight_xz = TransformerConv(input_size, hidden_size, bias=bias)
		self.weight_hz = TransformerConv(hidden_size, hidden_size, bias=bias)
		self.weight_xr = TransformerConv(input_size, hidden_size, bias=bias)
		self.weight_hr = TransformerConv(hidden_size, hidden_size, bias=bias)
		self.weight_xh = TransformerConv(input_size, hidden_size, bias=bias)
		self.weight_hh = TransformerConv(hidden_size, hidden_size, bias=bias)
	
	def forward(self, input, edge_index, h):
		z_g = torch.sigmoid(self.weight_xz(input, edge_index) + self.weight_hz(h, edge_index))
		r_g = torch.sigmoid(self.weight_xr(input, edge_index) + self.weight_hr(h, edge_index))
		h_tilde_g = torch.tanh(self.weight_xh(input, edge_index) + self.weight_hh(r_g * h, edge_index))
		h_out = z_g * h + (1 - z_g) * h_tilde_g	
		return h_out

def stack_truncate(Sample, Mask):
	# Initiate samples container
	Container = []
	# Permute dimensions (TxBxNxD -> BxTxNxD)
	Sample= Sample.permute(1,0,2,3)
	Mask = Mask.permute(1,0)
	for sample, mask in zip(Sample, Mask):
		# Truncate padding of T in each B
		sample = sample[mask]
		# Collect samples (B[TxNxD])
		Container.append(sample)
	# Container = torch.stack(Container, dim=1)
	return Container

#%%
class VGRNN(torch.nn.Module):
	def __init__(self, setting):
		super().__init__()

		model_params = setting['model_params']
		if 'recurrent' in setting: recurrent = setting['recurrent']
		else: recurrent = True
		if 'graphRNN' in setting: graphRNN = setting['graphRNN']
		else: graphRNN = True

		x_dim = model_params['x_dim']
		z_dim = model_params['z_dim']
		y_dim = model_params['y_dim']
		x_phi_dim = model_params['x_phi_dim']
		z_phi_dim = model_params['z_phi_dim']
		x_hidden_dim = model_params['x_hidden_dim']
		z_hidden_dim = model_params['z_hidden_dim']
		y_hidden_dim = model_params['y_hidden_dim']
		rnn_dim = model_params['rnn_dim']
		layer_dims = model_params['layer_dims']

		self.num_nodes = model_params['num_nodes']
		self.num_classes = model_params['num_classes']
		self.rnn_dim = rnn_dim
		self.recurrent = recurrent
		self.graphRNN = graphRNN
		self.device = get_device()
		self.EPS = 1e-15
		self.rng_state = torch.get_rng_state()

		# Data (x) extraction
		self.phi_x = dense_vary(x_dim, [], x_phi_dim, last_act='relu')

		# Latent state (z) prior
		self.prior_z_hidden = GCN_vary(rnn_dim, [], z_hidden_dim, last_act='relu')
		self.prior_z_mean = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
		self.prior_z_std = dense_vary(z_hidden_dim, [], z_dim, last_act='softplus')

		# Latent state (z) encoder
		if self.recurrent:
			enc_in_dim = x_phi_dim + rnn_dim
		else:
			enc_in_dim = x_phi_dim
		self.enc_z_hidden = GCN_vary(enc_in_dim, layer_dims, z_hidden_dim, last_act='relu')
		self.enc_z_mean = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
		self.enc_z_std = dense_vary(z_hidden_dim, [], z_dim, last_act='softplus')

		# Latent state (z) extraction
		self.phi_z = dense_vary(z_dim, [], z_phi_dim, last_act='relu')

		# Latent recurrent update
		if self.recurrent:
			rnn_in_dim = x_phi_dim + z_phi_dim
			if graphRNN: self.rnn_cell = GRU_GCN(rnn_in_dim, rnn_dim)
			else: self.rnn_cell = recurrent_cell(rnn_in_dim, rnn_dim, 'gru')

		# Graph (edge_index) decoder
		self.dec_graph = GraphDecoder()

		# Data (x) decoder
		if self.recurrent:
			dec_in_dim = z_phi_dim + rnn_dim
		else:
			dec_in_dim = z_phi_dim
		self.dec_x_hidden = GCN_vary(dec_in_dim, [], x_hidden_dim, last_act='relu')
		self.dec_x_mean = dense_vary(x_hidden_dim, [], x_dim, last_act=None)
		self.dec_x_std = dense_vary(x_hidden_dim, [], x_dim, last_act='softplus')

		# Graph classifier
		# readout_dim = self.num_nodes * (z_phi_dim)
		if self.recurrent:
			readout_dim = self.num_nodes * (z_phi_dim + rnn_dim)
		else:
			readout_dim = self.num_nodes * z_phi_dim
		self.classifier = GraphClassifier(readout_dim, y_hidden_dim, y_dim, dropout=0.5, batch_norm=True)

	def forward(self, graphs, setting, sample=False):

		variational = setting['variational']

		# Initiate containers
		Mask = []; Last = []
		x_NLL = []; z_KLD = []; adj_NLL = []
		Readout = []; Target= []

		if sample:
			z_Sample = []; adj_Sample = []; h_Sample = []; zh_Sample = []

		# Initiate rnn hidden
		batch_size = graphs[0].num_graphs
		h = torch.zeros(batch_size*self.num_nodes, self.rnn_dim).to(self.device, dtype=torch.float)

		for step, graph in enumerate(graphs):

			# Initiate graph variables
			x = graph.x.to(self.device)
			edge_index = graph.edge_index.to(self.device)
			batch = graph.batch.to(self.device)
			adj = graph.adj.to(self.device)
			y = graph.y.to(self.device)

			mask = (graph.pad.to(self.device) == False)
			last = graph.last.to(self.device)

			# Data (x) extraction
			x_phi = self.phi_x(x)

			# Latent state (z) encoder
			if self.recurrent: 
				enc_in = torch.cat([x_phi, h], dim=-1)
			else: 
				enc_in = x_phi
			z_enc_hidden = self.enc_z_hidden(enc_in, edge_index)
			z_enc_mean_sb = self.sep_batch(self.enc_z_mean(z_enc_hidden), batch)

			if variational:
				z_enc_std_sb = self.sep_batch(self.enc_z_std(z_enc_hidden), batch)
				# Latent state (z) prior
				if self.recurrent:
					z_prior_hidden = self.prior_z_hidden(h, edge_index)
					z_prior_mean_sb = self.sep_batch(self.prior_z_mean(z_prior_hidden), batch)
					z_prior_std_sb = self.sep_batch(self.prior_z_std(z_prior_hidden), batch)
				else:
					z_prior_mean_sb = torch.zeros(z_enc_mean_sb.shape).to(self.device)
					z_prior_std_sb = torch.zeros(z_enc_mean_sb.shape).to(self.device)
				# Latent state (z) kld loss
				z_kld_sb = self.kld_normal_sb(z_enc_mean_sb, z_enc_std_sb, z_prior_mean_sb, z_prior_std_sb, reduce=False)	
				# Latent state (z) repameterize
				if self.training:
					z_sample_sb = self.reparameterize_normal(z_enc_mean_sb, z_enc_std_sb)
				else:
					z_sample_sb = z_enc_mean_sb
			else:
				z_kld_sb = torch.zeros(batch_size).to(self.device)
				z_sample_sb = z_enc_mean_sb

			# Latent state (z) extraction
			z_phi = self.phi_z(torch.flatten(z_sample_sb, end_dim=1))

			# Graph (edge_index) decoder
			h_sb = self.sep_batch(h, batch)
			zh_sb = torch.cat([z_sample_sb, h_sb], dim=-1)
			if self.recurrent:
				adj_in_sb = zh_sb
			else:
				adj_in_sb = z_sample_sb
			adj_dec_sb = self.dec_graph.forward_sb(adj_in_sb)
			# Graph (edge_index) reconstruction loss
			adj_sb = self.sep_batch(adj, batch)
			adj_nll_sb = self.dec_graph.loss_sb(adj_dec_sb, adj_sb, reduce=False)

			# Latent recurrent update
			if self.recurrent:
				rnn_in = torch.cat([x_phi, z_phi], dim=-1)
				if self.graphRNN: 
					h = self.rnn_cell(rnn_in, edge_index, h)
				else: 
					h = self.rnn_cell(rnn_in, h)

			# # Data (x) decoder
			# dec_in = torch.cat([z_phi, h], dim=-1)
			# x_dec_hidden = self.dec_x_hidden(dec_in, edge_index)
			# x_dec_mean_sb = self.sep_batch(self.dec_x_mean(x_dec_hidden), batch)
			# x_dec_std_sb = self.sep_batch(self.dec_x_std(x_dec_hidden), batch)
			# # Data (x) nll loss
			# x_sb = self.sep_batch(x, batch)
			# x_nll_sb = self.nll_normal_sb(x_sb, x_dec_mean_sb, x_dec_std_sb, reduce=False)
			# # Data (x) repameterize
			# x_sample_sb = self.reparameterize_normal(x_dec_mean_sb, x_dec_std_sb)

			# Readout layer
			z_readout_sb = self.phi_z(z_enc_mean_sb)
			if self.recurrent:
				readout_sb = torch.cat([z_readout_sb, h_sb], dim=-1)
			else:
				readout_sb = z_readout_sb
			readout_flatten = readout_sb.flatten(start_dim=1, end_dim=2)

			# Containers append
			Mask.append(mask); Last.append(last)
			z_KLD.append(z_kld_sb); adj_NLL.append(adj_nll_sb)
			Readout.append(readout_flatten)
			Target.append(y)

			if sample: 
				z_Sample.append(z_sample_sb)
				adj_Sample.append(torch.sigmoid(adj_dec_sb))
				h_Sample.append(h_sb)
				zh_Sample.append(zh_sb)

		Mask = torch.stack(Mask)
		SeqLen = Mask.sum(dim=0)
		
		x_NLL = torch.zeros(1).to(self.device)
		z_KLD = ((torch.stack(z_KLD)*Mask).sum(dim=0) / SeqLen).mean()
		adj_NLL = ((torch.stack(adj_NLL)*Mask).sum(dim=0) / SeqLen).mean()

		Last = torch.stack(Last)
		assert Last.sum() == batch_size
		
		Readout = torch.stack(Readout)[Last]
		Target = torch.stack(Target)[Last]

		if sample: 
			z_Sample = stack_truncate(torch.stack(z_Sample), Mask)
			adj_Sample = stack_truncate(torch.stack(adj_Sample), Mask)
			h_Sample = stack_truncate(torch.stack(h_Sample), Mask)
			zh_Sample = stack_truncate(torch.stack(zh_Sample), Mask)
			return x_NLL, z_KLD, adj_NLL, Readout, Target, z_Sample, adj_Sample, h_Sample, zh_Sample
		else:
			return x_NLL, z_KLD, adj_NLL, Readout, Target

	def sep_batch(self, input, batch):
		output, _ = to_dense_batch(input, batch)
		return output

	def reparameterize_normal(self, mean, std):
		eps = torch.randn(mean.shape).to(self.device, dtype=torch.float)
		return mean + torch.mul(eps, std)

	def reparameterize_mvnormal(self, mean, node_sqm, feat_sqm):
		assert node_sqm.shape[0] == feat_sqm.shape[0]
		batch_size = node_sqm.shape[0]
		l_size = node_sqm.shape[-1]
		d_size = feat_sqm.shape[-2]
		eps = torch.randn(batch_size, l_size, d_size).to(self.device, dtype=torch.float)
		noise = node_sqm @ eps @ feat_sqm
		return torch.flatten(mean + noise, end_dim=1)

	def kld_normal_sb(self, mean_q, std_q, mean_p, std_p, reduce=False):
		kld_element = (2 * (torch.log(std_p + self.EPS) - torch.log(std_q + self.EPS)) +
					  (torch.pow(std_q + self.EPS , 2) + torch.pow(mean_q - mean_p, 2)) / 
					  torch.pow(std_p + self.EPS , 2) - 1)
		kld = (0.5 / self.num_nodes) * torch.sum(kld_element, dim=[-1,-2])
		if reduce: kld = torch.mean(kld)
		return kld

	def kld_mvnormal(self, mean_q, node_sqm_q, feat_std_q, mean_p, std_p):
		n_size, m_size = mean_q.shape[1:]

		node_cov_q = node_sqm_q @ torch.transpose(node_sqm_q, dim0=-2, dim1=-1)

		nll_q = n_size*(2*torch.log(feat_std_q + self.EPS)).sum(dim=-1) + m_size*torch.log(det(node_cov_q) + self.EPS) + n_size*m_size

		nll_p_det = (2*torch.log(std_p + self.EPS)).sum(dim=[-1,-2])

		UpVp = (std_p.pow(2) + self.EPS).pow(-1)
		UqVq = torch.diagonal(node_cov_q, dim1=-2, dim2=-1).unsqueeze(-1) @ feat_std_q.pow(2).unsqueeze(-2)
		nll_p_tr = (UqVq * UpVp).sum(dim=[-1,-2])

		Xd =  (mean_q - mean_p) * std_p
		nll_p_wls = (Xd * Xd).sum(dim=[-1,-2])

		nll_p = nll_p_det + nll_p_tr + nll_p_wls
		kld = (0.5 / (n_size*m_size)) * torch.mean(nll_p - nll_q, dim=0)
		
		# return kld
		return kld if kld > 0 else torch.zeros(1).to(self.device, dtype=torch.float)

	def nll_normal_sb(self, x, mean, std, reduce=False):
		constant = math.log(2*math.pi)
		xd = x - mean
		nll_element = 2*torch.log(std + self.EPS) + torch.div(xd, std + self.EPS).pow(2) + constant
		nll = (0.5 / (self.num_nodes * mean.shape[-1])) * torch.sum(nll_element, dim=[-1,-2])
		if reduce: nll = torch.mean(nll)
		return nll

	def edge_recon_loss(self, z, batch, pos_edge_index, neg_edge_index=None):
		num_pos_edges = pos_edge_index.shape[-1]
		pos_edges_dec = self.dec_graph(z, pos_edge_index)
		norm = (num_pos_edges * num_pos_edges) / float((num_pos_edges * num_pos_edges - pos_edges_dec.sum()) * 2)
		pos_loss = -torch.log(pos_edges_dec + self.EPS).mean()
		if neg_edge_index is None: neg_edge_index = batched_negative_sampling(pos_edge_index, batch)
		neg_edges_dec = self.dec_graph(z, neg_edge_index)
		neg_loss = - torch.log(1 - neg_edges_dec + self.EPS).mean()
		return norm * (pos_loss + neg_loss)
