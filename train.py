#%%
import os
import time
import pickle
import dill
import copy
import math
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

import matplotlib.pyplot as plt
from IPython.display import Audio, display, clear_output
from model import VGRNN

#%%
def save_obj(obj, path):
	if not path.endswith('.pkl'):
		path = path + '.pkl'
	with open(path, 'wb') as f:
		pickle.dump(obj, f)

def load_obj(path):
	if not path.endswith('.pkl'):
		path = path + '.pkl'
	with open(path, 'rb') as f:
		return pickle.load(f)

def check_path(path):
	if not path.endswith('.pkl'):
		path = path + '.pkl'
	return os.path.exists(path)

def load_data(outer_loop, inner_loop):
	saveTo = './folds_data/' 
	with open(saveTo+'graphs_outer'+str(outer_loop)+'_inner'+str(inner_loop)+'.pkl', 'rb') as f:
		f = torch.load(f)
		train_graphs = f['train_graphs']
		val_graphs = f['val_graphs']
		test_graphs = f['test_graphs']
	return train_graphs, val_graphs, test_graphs

class myDataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.num_node_features = data[0,0].num_node_features
		self.num_classes = data[0,0].num_classes
		self.num_nodes = data[0,0].num_nodes

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample

def padseq(batch):
	batch = np.asarray(batch).T
	if isinstance(batch[0,0], BaseData):
		new_batch = [Batch.from_data_list(graphs) for graphs in batch]
	return new_batch

def loadRNG(loadPATH):
	if os.path.exists(loadPATH):
		# print('loading rng state')
		checkpoint = torch.load(loadPATH)
		if 'rng_state' in checkpoint:
			torch.set_rng_state(checkpoint['rng_state'])

def loadCheckpoint(setting, loadPATH, savePATH):
	epochStart = 0
	train_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [],'y_BCE': [], 'y_ACC': [], 'Total': []}
	val_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [],'y_BCE': [], 'y_ACC': [], 'Total': []}
	test_losses = {'x_NLL': [], 'z_KLD': [], 'a_NLL': [],'y_BCE': [], 'y_ACC': [], 'Total': []}

	if 'rngPATH' in setting:
		loadRNG(setting['rngPATH'])

	model = VGRNN(setting)
	model.to(model.device)

	new_named_parameters = {}
	for key, item in model.named_parameters():
		if 'classifier' not in key:
			new_named_parameters[key] = item

	optimizers = []
	optimizers.append(optim.AdamW(new_named_parameters.values(), lr=setting['learnRate'][0], weight_decay=setting['l2factor'][0]))
	optimizers.append(optim.AdamW(model.classifier.parameters(), lr=setting['learnRate'][1], weight_decay=setting['l2factor'][1]))

	schedulers = []
	for i in range(2):
		if setting['lr_annealType'][i] == 'StepLR':
			schedulers.append(optim.lr_scheduler.StepLR(optimizers[i], step_size=2, gamma=0.96))
		elif setting['lr_annealType'][i] == 'ReduceLROnPlateau':
			schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], mode='min', patience=setting['lr_annealPatience'][i], factor=setting['lr_annealFactor'][i], verbose=False))
		elif setting['lr_annealType'][i] == None:
			schedulers.append(None)
		else: raise Exception("Learning rate annealing type not supported.")

	if os.path.exists(loadPATH):
		# print('load checkpoint')
		checkpoint = torch.load(loadPATH)
		model.load_state_dict(checkpoint['model_state_dict'])

		if loadPATH == savePATH:
			# print('loading training parameters')
			if checkpoint['epoch'] == 0:
				epochStart = checkpoint['epoch']
			else:
				epochStart = checkpoint['epoch'] + 1
			train_losses = checkpoint['train_losses']
			val_losses = checkpoint['val_losses']
			test_losses = checkpoint['test_losses']
			
			for i in range(2):
				optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
				if schedulers[i] is not None and len(checkpoint['scheduler_state_dicts'][i]) != 0:
					schedulers[i].load_state_dict(checkpoint['scheduler_state_dicts'][i])
			# Reset learning rate
			# optimizer.param_groups[0]['lr'] = setting['learnRate']

	else:
		# print('new checkpoint')
		model_state_dict = copy.deepcopy(model.state_dict())

		optimizer_state_dicts = []
		scheduler_state_dicts = []
		for i in range(2):
			optimizer_state_dicts.append(copy.deepcopy(optimizers[i].state_dict()))
			if schedulers[i] is not None:
				scheduler_state_dicts.append(copy.deepcopy(schedulers[i].state_dict()))
			else:
				scheduler_state_dicts.append(None)

		torch.save({'model_state_dict': model_state_dict,
					'optimizer_state_dicts': optimizer_state_dicts,
					'scheduler_state_dicts': scheduler_state_dicts,
					'train_losses': train_losses,
					'val_losses': val_losses,
					'test_losses': test_losses,
					'epoch': 0,
					'training_setting': setting
					}, savePATH)

	return model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses


def train(model, optimizers, schedulers, setting, checkpointPATH,
		  train_losses, val_losses, test_losses, 
		  train_loader, val_loader, test_loader,
		  epochStart=0, numEpochs=100, gradThreshold=1, gradientClip=True,
		  verboseFreq=1, verbose=True, valFreq=0, 
		  validation=False, testing=False, 
		  earlyStopPatience=1, earlyStop=True):
	
	print('Current device: %s' % model.device)
	# torch.autograd.set_detect_anomaly(True)

	# Initiate training and validation losses
	x_NLL, z_KLD, a_NLL, y_BCE, y_ACC, total_loss = (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

	# Get number of validation batches/iterations
	# numIterVal = len(val_loader)

	# Initiate validation and testing losses
	valLoss = {'x_NLL': torch.zeros(1).to(model.device),
			   'z_KLD': torch.zeros(1).to(model.device),
			   'a_NLL': torch.zeros(1).to(model.device),
			   'y_BCE': torch.zeros(1).to(model.device),
			   'y_ACC': torch.zeros(1).to(model.device),
			   'Total': torch.zeros(1).to(model.device)}

	testLoss = {'x_NLL': torch.zeros(1).to(model.device),
			   'z_KLD': torch.zeros(1).to(model.device),
			   'a_NLL': torch.zeros(1).to(model.device),
			   'y_BCE': torch.zeros(1).to(model.device),
			   'y_ACC': torch.zeros(1).to(model.device),
			   'Total': torch.zeros(1).to(model.device)}

	# Initiate best model params, best validation loss (large value) and patience counter
	best_params = {}
	best_valLoss = math.inf
	best_testLoss = math.inf
	best_atEpoch = 0
	patience_count = 0
	patience_metric = 0
	lr_anneal_metric = 0
	num_bad_epochs = [None, None]
	Print = None

	# Initiate stopTraining as False
	# stopTraining = False

	# Start time
	start = time.time()

	for epoch in range(epochStart, numEpochs):

		# Reset epoch training and validation losses
		trainLoss = {'x_NLL': torch.zeros(1).to(model.device),
					 'z_KLD': torch.zeros(1).to(model.device),
					 'a_NLL': torch.zeros(1).to(model.device),
					 'y_BCE': torch.zeros(1).to(model.device),
					 'y_ACC': torch.zeros(1).to(model.device),
					 'Total': torch.zeros(1).to(model.device)}

		numIter = len(train_loader)

		for idxTrain, batch in enumerate(train_loader):

			model.train()

			# Joint Training
			for i, (optimizer, multiplier) in enumerate(zip(optimizers, setting['yBCEMultiplier'])):
				optimizer.zero_grad()

				# Compute batch training losses
				if i == 0:
					x_NLL, z_KLD, a_NLL, Readout, Target = model(batch, setting, sample=False)
					y_BCE, y_ACC = model.classifier(Readout, Target, sample=False)
					loss = x_NLL + z_KLD + a_NLL
					if setting['DAT']:
						loss = loss - multiplier*y_BCE
				else:
					y_BCE, y_ACC = model.classifier(Readout.detach(), Target, sample=False)
					loss = multiplier*y_BCE

				# Backpropagate
				loss.backward()
			
				# Gradient clipping
				if gradientClip: nn.utils.clip_grad_norm_(model.parameters(), gradThreshold, norm_type=2, error_if_nonfinite=False)

				# Gradient descent
				optimizer.step()

			with torch.no_grad():
				total_loss = x_NLL + z_KLD + a_NLL + y_BCE
				trainLoss['x_NLL'] += x_NLL / numIter
				trainLoss['z_KLD'] += z_KLD / numIter
				trainLoss['a_NLL'] += a_NLL / numIter
				trainLoss['y_BCE'] += y_BCE / numIter
				trainLoss['y_ACC'] += y_ACC / numIter
				trainLoss['Total'] += total_loss / numIter

			if verbose and (idxTrain+1 == 1 or (idxTrain+1) % verboseFreq == 0):
				current = time.time()

				Print = 'Time Elapsed: %d  Epoch: %d/%d  Iteration: %d/%d  Sequence Length: %s  Batch Size: %s \n'\
				'Variational Bayes: %s  Domain Adversarial: %s  Graph RNN: %s  Recurrent: %s \n'\
				'Learning Rates: %.2e, %.2e  BCE Multipliers: %.0e, %.0e  Anneal Metric: %.4f \n'\
				'Patience: %d/%d  No. of Bad Epochs: %s, %s  Patience Metric: %.4f \n'\
				'Best Validation Loss = %.4f  Testing Loss at Best Validation = %.4f  Best at Epoch: %d \n'\
				'Training -- x_NLL = %.4f  z_KLD = %.4f  a_NLL = %.4f  y_BCE = %.4f  y_ACC = %.4f  Total = %.4f \n'\
				'Validation -- x_NLL = %.4f  z_KLD = %.4f  a_NLL = %.4f  y_BCE = %.4f  y_ACC = %.4f  Total = %.4f \n'\
				'Testing -- x_NLL = %.4f  z_KLD = %.4f  a_NLL = %.4f  y_BCE = %.4f  y_ACC = %.4f  Total = %.4f \n'\
				%(current-start, epoch+1, numEpochs, idxTrain+1, numIter, len(batch), batch[0].num_graphs, 
				setting['variational'], setting['DAT'], setting['graphRNN'], setting['recurrent'],
				optimizers[0].param_groups[0]['lr'], optimizers[1].param_groups[0]['lr'], setting['yBCEMultiplier'][0], setting['yBCEMultiplier'][1], lr_anneal_metric,
				patience_count, earlyStopPatience, num_bad_epochs[0], num_bad_epochs[1], patience_metric,
				best_valLoss, best_testLoss, best_atEpoch,
				x_NLL, z_KLD, a_NLL, y_BCE, y_ACC, total_loss,
				valLoss['x_NLL'], valLoss['z_KLD'], valLoss['a_NLL'], valLoss['y_BCE'], valLoss['y_ACC'], valLoss['Total'], 
				testLoss['x_NLL'], testLoss['z_KLD'], testLoss['a_NLL'], testLoss['y_BCE'], testLoss['y_ACC'], testLoss['Total'])

				clear_output(wait=True)
				print(Print)

		# Collect epoch training losses
		train_losses['x_NLL'].append(trainLoss['x_NLL'])
		train_losses['z_KLD'].append(trainLoss['z_KLD'])
		train_losses['a_NLL'].append(trainLoss['a_NLL'])
		train_losses['y_BCE'].append(trainLoss['y_BCE'])
		train_losses['y_ACC'].append(trainLoss['y_ACC'])
		train_losses['Total'].append(trainLoss['Total'])

		if (epoch+1) % valFreq == 0:
			if validation:
				# Compute batch validation losses
				valLoss = validate(model, setting, val_loader)
				# Collect epoch validation losses
				val_losses['x_NLL'].append(valLoss['x_NLL'])
				val_losses['z_KLD'].append(valLoss['z_KLD'])
				val_losses['a_NLL'].append(valLoss['a_NLL'])
				val_losses['y_BCE'].append(valLoss['y_BCE'])
				val_losses['y_ACC'].append(valLoss['y_ACC'])
				val_losses['Total'].append(valLoss['Total'])

			if testing:
				# Compute batch validation losses
				testLoss = validate(model, setting, test_loader)
				# Collect epoch validation losses
				test_losses['x_NLL'].append(testLoss['x_NLL'])
				test_losses['z_KLD'].append(testLoss['z_KLD'])
				test_losses['a_NLL'].append(testLoss['a_NLL'])
				test_losses['y_BCE'].append(testLoss['y_BCE'])
				test_losses['y_ACC'].append(testLoss['y_ACC'])
				test_losses['Total'].append(testLoss['Total'])
				# Early stop patience metric
				test_metric = testLoss['y_ACC']
		
			if earlyStop:
				# Early stop patience metric
				patience_metric = trainLoss['y_BCE'] + trainLoss['y_ACC']
				# Save model params with best validation loss
				if (patience_metric <= best_valLoss):
					best_valLoss = patience_metric
					best_testLoss = test_metric
					best_params = copy.deepcopy(model.state_dict())
					best_atEpoch = epoch + 1
					patience_count = 0
				else:
					patience_count += 1
				# print('Patience: %d/%d' %(patience_count,earlyStopPatience))
			else:
				best_params = copy.deepcopy(model.state_dict())
		
		# Learning rate annealing
		lr_anneal_metric = trainLoss['y_BCE']
		for i, scheduler in enumerate(schedulers):
			if type(scheduler) == torch.optim.lr_scheduler.StepLR:
				scheduler.step()
			elif type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
				scheduler.step(lr_anneal_metric)
				num_bad_epochs[i] = scheduler.state_dict()['num_bad_epochs']
			elif type(scheduler) == type(None): pass	
			else: raise Exception("Learning rate annealing type not supported.")

		optimizer_state_dicts = []
		scheduler_state_dicts = []
		for i in range(2):
			optimizer_state_dicts.append(copy.deepcopy(optimizers[i].state_dict()))
			if schedulers[i] is not None:
				scheduler_state_dicts.append(copy.deepcopy(schedulers[i].state_dict()))
			else:
				scheduler_state_dicts.append(None)

		# Save training checkpoint after every epoch
		torch.save({'model_state_dict': best_params,
					'optimizer_state_dicts': optimizer_state_dicts,
					'scheduler_state_dicts': scheduler_state_dicts,
					'train_losses': train_losses,
					'val_losses': val_losses,
					'test_losses': test_losses,
					'epoch': epoch,
					'training_setting': setting,
					'rng_state': model.rng_state
					}, checkpointPATH)

		# Early stopping (break out of epoch loop)
		if earlyStop and patience_count > earlyStopPatience:
			print('Early Stopped.')
			# break from epoch loop
			break

	# Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg", autoplay=True)
	print('Training done!')
	return model, train_losses, val_losses, test_losses


def validate(model, setting, val_loader):
	# try:
	# 	# Try to get next item from validation iterator
	# 	tup = next(valIterator)
	# 	# Increment validation index
	# 	idxVal += 1
	# # Except error when validation iterator runs out
	# except:
	# 	# Reset validation losses
	# 	valLoss = {'NLLUx': 0,'NLLUz': 0,'NLLSz': 0,'NLLSu': 0,'Total': 0}
	# 	# Reset validation iterator 
	# 	valIterator = iter(val_loader)
	# 	# Get next item from validation iterator
	# 	tup = next(valIterator)
	# 	# Reset validation index
	# 	idxVal = 0
	# Get number of validation batches/iterations
	# numIter = len(valIterator)

	# Get number of validation batches/iterations
	numIter = len(val_loader)
	# Reset validation losses
	valLoss = {'x_NLL': torch.zeros(1).to(model.device),
			   'z_KLD': torch.zeros(1).to(model.device),
			   'a_NLL': torch.zeros(1).to(model.device),
			   'y_BCE': torch.zeros(1).to(model.device),
			   'y_ACC': torch.zeros(1).to(model.device),
			   'Total': torch.zeros(1).to(model.device)}

	for idxVal, batch in enumerate(val_loader):

		# Compute validation batch losses (and generate validation samples)
		model.eval()
		with torch.no_grad():
			x_NLL, z_KLD, a_NLL, Readout, Target = model(batch, setting, sample=False)
			y_BCE, y_ACC = model.classifier(Readout, Target, sample=False)
			total_loss = x_NLL + z_KLD + a_NLL + y_BCE

		# Average batch validation losses per epoch
		with torch.no_grad():
			valLoss['x_NLL'] += x_NLL / numIter
			valLoss['z_KLD'] += z_KLD / numIter
			valLoss['a_NLL'] += a_NLL / numIter
			valLoss['y_BCE'] += y_BCE / numIter
			valLoss['y_ACC'] += y_ACC / numIter
			valLoss['Total'] += total_loss / numIter

	return valLoss
# %%
