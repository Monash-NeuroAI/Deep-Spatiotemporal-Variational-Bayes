#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

from model import *
from train import *

def add(x,y):
	return x+y

outer_loop = 1; inner_loop = 1
train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)

train_dataset = myDataset(np.concatenate([train_graphs, val_graphs], axis=0))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

val_dataset = myDataset(val_graphs)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

test_dataset = myDataset(test_graphs)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)

partition = [len(train_dataset), len(val_dataset), len(test_dataset)]
print(len(train_graphs), len(val_graphs), len(test_graphs))
print(partition)

# %%
# Initialize training
savePATH = '/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold'\
+str(outer_loop)+str(inner_loop)+'_16.10_v-h'
loadPATH = savePATH
# loadPATH = savePATH
# /home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/ARCHIVE/VGRNN_softmax_adv_fold21_v14.50

model_params = {'num_nodes':264, 'num_classes':2, 'x_dim':264, 'y_dim':2,
				'z_hidden_dim':32, 'z_dim':16, 'z_phi_dim':8,
				'x_phi_dim':64, 'rnn_dim':16, 'y_hidden_dim':[32],
				'x_hidden_dim':64, 'layer_dims':[]
				}

# 4096,2048,1024,512,256,128,64,32,16,8,4,2

lr_annealType = 'ReduceLROnPlateau'
lr_annealType = [lr_annealType, lr_annealType]

setting = {
'rngPATH': r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_vdh",
'model_params': model_params,
'recurrent': True,
'learnRate': [0.00001, 0.00001],
'yBCEMultiplier': [1, 1],
'l2factor': [0.01, 0.01],
'lr_annealType': lr_annealType,
'lr_annealFactor': [0.5, 0.5],
'lr_annealPatience': [20, 20],
'variational': True,
'DAT': False,
'graphRNN': True,
'partition': partition
}

# Acquire model, optimizer and scheduler
model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)
print(model)
print(model_params)
for optimizer in optimizers:
	print(optimizer)
for scheduler in schedulers:
	print(scheduler)
print(setting['rngPATH'])
print(savePATH)

# %%
# Train model
model, train_losses, val_losses, test_losses = train(
model, optimizers, schedulers, setting, savePATH,
train_losses, val_losses, test_losses,
train_loader, val_loader, test_loader, 
epochStart=epochStart, numEpochs=500, 
gradThreshold=1, gradientClip=True,
verboseFreq=1, verbose=True, valFreq=1, 
validation=False, testing=True,
earlyStopPatience=500, earlyStop=True)

# %%
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.2_---"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.1_---"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.1_---"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.1_---"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_---"

# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.0_--h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_--h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.1_--h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_--h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_--h"

# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.3_-dh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.3_-dh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.3_-dh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.3_-dh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.3_-dh"

# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.2_v-h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_v-h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.0_v-h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_v-h"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_v-h"

# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.2_vdh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_vdh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.1_vdh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.1_vdh"
# savePATH = r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_vdh"

loadPATH = savePATH
checkpoint = torch.load(savePATH)
setting = checkpoint['training_setting']

plt.plot(torch.stack(checkpoint['train_losses']['z_KLD'], dim=0).cpu().numpy())
# plt.show()
# plt.plot(torch.stack(checkpoint['val_losses']['z_KLD'], dim=0).cpu().numpy())
# plt.show()
plt.plot(torch.stack(checkpoint['test_losses']['z_KLD'], dim=0).cpu().numpy())
plt.legend(['train','val','test'])
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['a_NLL'], dim=0).cpu().numpy())
# plt.show()
# plt.plot(torch.stack(checkpoint['val_losses']['a_NLL'], dim=0).cpu().numpy())
# plt.show()
plt.plot(torch.stack(checkpoint['test_losses']['a_NLL'], dim=0).cpu().numpy())
plt.legend(['train','val','test'])
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['y_BCE'], dim=0).cpu().numpy())
# plt.show()
# plt.plot(torch.stack(checkpoint['val_losses']['y_BCE'], dim=0).cpu().numpy())
# plt.show()
plt.plot(torch.stack(checkpoint['test_losses']['y_BCE'], dim=0).cpu().numpy())
# plt.legend(['train','val','test'])
plt.show()

plt.plot(torch.stack(checkpoint['train_losses']['y_ACC'], dim=0).cpu().numpy())
# plt.show()
# plt.plot(torch.stack(checkpoint['val_losses']['y_ACC'], dim=0).cpu().numpy())
# plt.show()
plt.plot(torch.stack(checkpoint['test_losses']['y_ACC'], dim=0).cpu().numpy())
# plt.legend(['train','val','test'])
plt.show()

print(savePATH)
print(checkpoint['training_setting'])
print(checkpoint['test_losses']['y_BCE'][-1], 1 - checkpoint['test_losses']['y_ACC'][-1])

# %%
savePATHS = []; test_ACC = []; test_PRED = []; test_SENS = []; test_F1 = []; test_AUC = []
z_Samples = []; zh_Samples = []; Readouts = []; Targets = []; testIdx = []

# title = "Graph Autoencoder"
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.0_ae")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_ae")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.0_ae")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_ae")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.0_ae")

# title = "Graph Autoencoder + Conventional GRU"
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.0_---")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_---")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_---")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.4_---")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.4_---")

title = "Graph Autoencoder + Spatial-aware GRU"
savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.3_--h")
savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_--h")
savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.1_--h")
savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_--h")
savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_--h")

# title = "Graph Autoencoder + Spatial-aware GRU + Adverserial Regularization"
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.3_-dh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_-dh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_-dh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.4_-dh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.3_-dh")

# title = "Variational Graph Autoencoder + Spatial-aware GRU"
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.4_v-h")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.1_v-h")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.0_v-h")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_v-h")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_v-h")

# title = "Variational Graph Autoencoder + Spatial-aware GRU + Adverserial Regularization"
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.4_vdh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_vdh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_vdh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.1_vdh")
# savePATHS.append(r"/home/admin01/Junn/TCM/VGRN_Junn/Junn/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_vdh")

for savePATH in savePATHS:

	outer_loop = list(savePATH.split('_')[-3])[-2]
	train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)

	dataset = myDataset(np.concatenate([train_graphs, val_graphs, test_graphs], axis=0))
	# dataset = myDataset(np.concatenate([test_graphs], axis=0))
	dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, collate_fn=padseq, pin_memory=True)

	test_idx = torch.zeros(len(dataset))
	test_idx[:len(test_graphs)] = True

	loadPATH = savePATH
	checkpoint = torch.load(savePATH)
	setting = checkpoint['training_setting']
	model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)

	for idxVal, batch in enumerate(dataloader):

		# Compute validation batch losses (and generate validation samples)
		model.eval()
		with torch.no_grad():
			x_NLL, z_KLD, a_NLL, Readout, Target, z_Sample, a_Sample, h_Sample, zh_Sample = model(batch, setting, sample=True)
			y_BCE, y_ACC, y_Prob, y_Pred = model.classifier(Readout, Target, sample=True)
			total_loss = x_NLL + z_KLD + a_NLL + y_BCE

	print('outer_loop: '+ str(outer_loop))
	print(savePATH)
	print(setting['recurrent'] if 'recurrent' in setting else True, setting['variational'], setting['DAT'], setting['graphRNN'] if 'graphRNN' in setting else True)
	# print(y_BCE, 1-y_ACC)

	z_Samples.append(z_Sample)
	zh_Samples.append(zh_Sample)
	Readouts.append(Readout)
	Targets.append(Target)
	testIdx.append(test_idx)

# 	Pred = y_Pred.detach().cpu().numpy()
# 	Actual = Target.detach().cpu().numpy()
# 	# accuracy
# 	test_acc = np.sum(Pred==Actual)/len(Pred)
# 	test_ACC.append(test_acc)
# 	# precision
# 	test_pred = precision_score(Actual, Pred)
# 	test_PRED.append(test_pred)
# 	# recall
# 	test_sens = recall_score(Actual, Pred)
# 	test_SENS.append(test_sens)
# 	# F1-score
# 	test_f1 = f1_score(Actual, Pred)
# 	test_F1.append(test_f1)
# 	# AUC
# 	test_auc = roc_auc_score(Actual, Pred)
# 	test_AUC.append(test_auc)
# 	print(test_acc, test_pred, test_sens, test_f1, test_auc)

# print(np.mean(test_ACC), np.std(test_ACC))
# print(np.mean(test_PRED), np.std(test_PRED))
# print(np.mean(test_SENS), np.std(test_SENS))
# print(np.mean(test_F1), np.std(test_F1))
# print(np.mean(test_AUC), np.std(test_AUC))

## %%
# saveTo = './samples_data/'  
# os.makedirs(saveTo, exist_ok=True)

# samples = {}
# samples['PATH'] = savePATH
# samples['metrics'] = (y_BCE.detach().cpu().numpy(), y_ACC.detach().cpu().numpy())
# samples['latent'] = [sample.detach().cpu().numpy() for sample in z_Sample]
# samples['adjacency'] = [sample.detach().cpu().numpy() for sample in a_Sample]
# samples['recurrent'] = [sample.detach().cpu().numpy() for sample in h_Sample]
# samples['embedding'] = [sample.detach().cpu().numpy() for sample in zh_Sample]
# samples['target'] = Target.detach().cpu().numpy()
# samples['prob'] = y_Prob.detach().cpu().numpy()
# samples['pred'] = y_Pred.detach().cpu().numpy()

# print('Saving samples ...')     
# with open(saveTo+'samples_outer'+str(outer_loop)+'.pkl', 'wb') as f:
# 	torch.save(samples, f)
# 	f.close()

# %%
plt.style.use('default')
plt.style.use('seaborn-darkgrid')
init = 'default'
tsne = TSNE(n_components=2)

Target = torch.cat(Targets).detach().cpu().numpy()
hc_idx = (Target == 0)
asd_idx = (Target == 1)

test_idx = torch.cat(testIdx).detach().cpu().numpy()
train_idx = (test_idx == 0)
test_idx = (test_idx == 1)

samples = [item.detach().cpu().numpy() for sublist in z_Samples for item in sublist]
data_fit = [sample.reshape(len(sample),-1) for sample in samples]
sub_len = [len(data) for data in data_fit]
split_idx = np.cumsum(sub_len)
tsne_embed = tsne.fit_transform(np.concatenate(data_fit, axis=0))
z_embed = np.split(tsne_embed, split_idx, axis=0)[:-1]
print(sub_len == [len(item) for item in z_embed])
z_embed_hc_train = np.concatenate([z_embed[i] for i in np.where(hc_idx*train_idx)[0]], axis=0)
z_embed_asd_train = np.concatenate([z_embed[i] for i in np.where(asd_idx*train_idx)[0]], axis=0)
z_embed_hc_test = np.concatenate([z_embed[i] for i in np.where(hc_idx*test_idx)[0]], axis=0)
z_embed_asd_test = np.concatenate([z_embed[i] for i in np.where(asd_idx*test_idx)[0]], axis=0)

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.scatter(z_embed_hc_train[:,0], z_embed_hc_train[:,1], c='blue', marker='.', s=40, linewidth=1)
ax.scatter(z_embed_asd_train[:,0], z_embed_asd_train[:,1], c='blue', marker='x', s=40, linewidth=1)
ax.scatter(z_embed_hc_test[:,0], z_embed_hc_test[:,1], c='red', marker='.', s=40, linewidth=1)
ax.scatter(z_embed_asd_test[:,0], z_embed_asd_test[:,1], c='red', marker='x', s=40, linewidth=1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Readout = torch.cat(Readouts).detach().cpu().numpy()
# r_embed = tsne.fit_transform(Readout)
# r_embed_hc_train = r_embed[hc_idx*train_idx]
# r_embed_asd_train = r_embed[asd_idx*train_idx]
# r_embed_hc_test = r_embed[hc_idx*test_idx]
# r_embed_asd_test = r_embed[asd_idx*test_idx]

# fig = plt.figure(figsize=(10,10))
# ax = plt.axes()
# ax.scatter(r_embed_hc_train[:,0], r_embed_hc_train[:,1], c='blue', marker='.', s=80, linewidth=2)
# ax.scatter(r_embed_asd_train[:,0], r_embed_asd_train[:,1], c='blue', marker='x', s=80, linewidth=2)
# ax.scatter(r_embed_hc_test[:,0], r_embed_hc_test[:,1], c='red', marker='.', s=80, linewidth=2)
# ax.scatter(r_embed_asd_test[:,0], r_embed_asd_test[:,1], c='red', marker='x', s=80, linewidth=2)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

saveTo = './saved_figures/'  
os.makedirs(saveTo, exist_ok=True)
# plt.title(title)
# plt.legend(['HC train','ASD train','HC test','ASD test'], loc='best', ncol=1, columnspacing=0.3, handletextpad=0.3, borderpad=0.2, fontsize=20)
plt.savefig(saveTo+title+'_Z'+'_'+init+'.jpg', dpi=400, bbox_inches='tight', pad_inches=0)
plt.show()

# %%
# dataset = myDataset(np.concatenate([test_dataset], axis=0))
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0, collate_fn=padseq, pin_memory=True)
# self = model.eval()
# sample = True

# for idx, graphs in enumerate(dataloader):

# 	variational = setting['variational']

# 	# Initiate containers
# 	Mask = []; Last = []
# 	x_NLL = []; z_KLD = []; adj_NLL = []
# 	# y_BCE = []; y_ACC = []
# 	Readout = []; Target= []
# 	# x_NLL = torch.zeros(1).to(self.device); z_KLD = torch.zeros(1).to(self.device)
# 	# adj_NLL = torch.zeros(1).to(self.device); y_BCE = torch.zeros(1).to(self.device)
# 	if sample:
# 		z_Sample = []; adj_Sample = []; h_Sample = []
# 		# x_Sample = []; y_Sample = []

# 	# Initiate rnn hidden
# 	batch_size = graphs[0].num_graphs
# 	h = torch.zeros(batch_size*self.num_nodes, self.rnn_dim).to(self.device, dtype=torch.float)

# 	for step, graph in enumerate(graphs):

# 		# Initiate graph variables
# 		x = graph.x.to(self.device)
# 		edge_index = graph.edge_index.to(self.device)
# 		batch = graph.batch.to(self.device)
# 		adj = graph.adj.to(self.device)
# 		y = graph.y.to(self.device)

# 		mask = (graph.pad.to(self.device) == False)
# 		last = graph.last.to(self.device)

# 		# Data (x) extraction
# 		x_phi = self.phi_x(x)

# 		# Latent state (z) encoder
# 		enc_in = torch.cat([x_phi, h], dim=-1)
# 		z_enc_hidden = self.enc_z_hidden(enc_in, edge_index)
# 		z_enc_mean_sb = self.sep_batch(self.enc_z_mean(z_enc_hidden), batch)

# 		if variational:
# 			z_enc_std_sb = self.sep_batch(self.enc_z_std(z_enc_hidden), batch)
# 			# Latent state (z) prior
# 			z_prior_hidden = self.prior_z_hidden(h, edge_index)
# 			z_prior_mean_sb = self.sep_batch(self.prior_z_mean(z_prior_hidden), batch)
# 			z_prior_std_sb = self.sep_batch(self.prior_z_std(z_prior_hidden), batch)
# 			# Latent state (z) kld loss
# 			z_kld_sb = self.kld_normal_sb(z_enc_mean_sb, z_enc_std_sb, z_prior_mean_sb, z_prior_std_sb, reduce=False)	
# 			# Latent state (z) repameterize
# 			if self.training:
# 				z_sample_sb = self.reparameterize_normal(z_enc_mean_sb, z_enc_std_sb)
# 			else:
# 				z_sample_sb = z_enc_mean_sb
# 		else:
# 			z_kld_sb = torch.zeros(batch_size).to(self.device)
# 			z_sample_sb = z_enc_mean_sb

# 		# Latent state (z) extraction
# 		z_phi = self.phi_z(torch.flatten(z_sample_sb, end_dim=1))

# 		# Graph (edge_index) decoder
# 		h_sb = self.sep_batch(h, batch)
# 		adj_in_sb = torch.cat([z_sample_sb, h_sb], dim=-1)
# 		adj_dec_sb = self.dec_graph.forward_sb(adj_in_sb)
# 		# Graph (edge_index) reconstruction loss
# 		adj_sb = self.sep_batch(adj, batch)
# 		adj_nll_sb = self.dec_graph.loss_sb(adj_dec_sb, adj_sb, reduce=False)

# 		# Latent recurrent update
# 		rnn_in = torch.cat([x_phi, z_phi], dim=-1)
# 		h = self.rnn_cell(rnn_in, edge_index, h)

# 		# # Data (x) decoder
# 		# dec_in = torch.cat([z_phi, h], dim=-1)
# 		# x_dec_hidden = self.dec_x_hidden(dec_in, edge_index)
# 		# x_dec_mean_sb = self.sep_batch(self.dec_x_mean(x_dec_hidden), batch)
# 		# x_dec_std_sb = self.sep_batch(self.dec_x_std(x_dec_hidden), batch)
# 		# # Data (x) nll loss
# 		# x_sb = self.sep_batch(x, batch)
# 		# x_nll_sb = self.nll_normal_sb(x_sb, x_dec_mean_sb, x_dec_std_sb, reduce=False)
# 		# # Data (x) repameterize
# 		# x_sample_sb = self.reparameterize_normal(x_dec_mean_sb, x_dec_std_sb)

# 		# Readout layer
# 		# z_readout = z_phi
# 		# z_readout = self.phi_z(torch.flatten(z_enc_mean_sb, end_dim=1))
# 		# readout = torch.cat([z_readout, h], dim=-1)
# 		# readout_sb = self.sep_batch(readout, batch)
# 		readout_sb = torch.cat([z_enc_mean_sb, h_sb], dim=-1)
# 		readout_flatten = readout_sb.flatten(start_dim=1, end_dim=2)
# 		# readout_in = global_mean_pool(readout_in, batch)
# 		# Graph Classifier
# 		# y_bce_sb, y_acc_sb, y_sample_sb = self.classifier(readout_in_sb, y)

# 		# Containers append
# 		Mask.append(mask); Last.append(last)
# 		# x_NLL.append(x_nll_sb*mask)
# 		z_KLD.append(z_kld_sb); adj_NLL.append(adj_nll_sb)
# 		# y_BCE.append(y_bce_sb), y_ACC.append(y_acc_sb)
# 		Readout.append(readout_flatten)
# 		Target.append(y)

# 		if sample: 
# 			# x_Sample.append(x_sample_sb*mask)
# 			z_Sample.append(z_sample_sb)
# 			adj_Sample.append(torch.sigmoid(adj_dec_sb))
# 			h_Sample.append(h_sb)
# 			# y_Sample.append(y_sample_sb*mask)

# 	Mask = torch.stack(Mask)
# 	SeqLen = Mask.sum(dim=0)
# 	# x_NLL = ((torch.stack(x_NLL)*Mask).sum(dim=0) / SeqLen).mean()
# 	x_NLL = torch.zeros(1).to(self.device)
# 	z_KLD = ((torch.stack(z_KLD)*Mask).sum(dim=0) / SeqLen).mean()
# 	adj_NLL = ((torch.stack(adj_NLL)*Mask).sum(dim=0) / SeqLen).mean()

# 	Last = torch.stack(Last)
# 	assert Last.sum() == batch_size
# 	# y_BCE = torch.stack(y_BCE)[Last].mean()
# 	# y_ACC = torch.stack(y_ACC)[Last].mean()
# 	Readout = torch.stack(Readout)[Last]
# 	Target = torch.stack(Target)[Last]
