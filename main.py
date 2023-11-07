#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from model import *
from train import *

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

model_params = {'num_nodes':264, 'num_classes':2, 'x_dim':264, 'y_dim':2,
				'z_hidden_dim':32, 'z_dim':16, 'z_phi_dim':8,
				'x_phi_dim':64, 'rnn_dim':16, 'y_hidden_dim':[32],
				'x_hidden_dim':64, 'layer_dims':[]
				}

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

# Assign the value of 'savePATH' to 'loadPATH'
loadPATH = savePATH

# Load a PyTorch checkpoint from the file specified by 'savePATH'
checkpoint = torch.load(savePATH)

# Extract the training settings from the loaded checkpoint
setting = checkpoint['training_setting']

# Plot the training and validation/test losses for the z_KLD (Kullback-Leibler Divergence) term
plt.plot(torch.stack(checkpoint['train_losses']['z_KLD'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['z_KLD'], dim=0).cpu().numpy())
plt.legend(['train', 'val', 'test'])  # Add a legend to the plot
plt.show()  # Display the plot

# Plot the training and validation/test losses for the a_NLL (Negative Log Likelihood) term
plt.plot(torch.stack(checkpoint['train_losses']['a_NLL'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['a_NLL'], dim=0).cpu().numpy())
plt.legend(['train', 'val', 'test'])  # Add a legend to the plot
plt.show()  # Display the plot

# Plot the training and validation/test losses for the y_BCE (Binary Cross-Entropy) term
plt.plot(torch.stack(checkpoint['train_losses']['y_BCE'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['y_BCE'], dim=0).cpu().numpy())
plt.show()  # Display the plot

# Plot the training and validation/test losses for the y_ACC (Accuracy) term
plt.plot(torch.stack(checkpoint['train_losses']['y_ACC'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['y_ACC'], dim=0).cpu().numpy())
plt.show()  # Display the plot

print(savePATH)
print(checkpoint['training_setting'])
print(checkpoint['test_losses']['y_BCE'][-1], 1 - checkpoint['test_losses']['y_ACC'][-1])

# %%
savePATHS = []; test_ACC = []; test_PRED = []; test_SENS = []; test_F1 = []; test_AUC = []
z_Samples = []; zh_Samples = []; Readouts = []; Targets = []; testIdx = []

# FOR ABLATION STUDY

# title = "Graph Autoencoder"
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.0_ae")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_ae")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.0_ae")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_ae")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.0_ae")

# title = "Graph Autoencoder + Conventional GRU"
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.0_---")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_---")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_---")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.4_---")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.4_---")

title = "Graph Autoencoder + Spatial-aware GRU"
savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.3_--h")
savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_--h")
savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.1_--h")
savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_--h")
savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_--h")

# title = "Graph Autoencoder + Spatial-aware GRU + Adverserial Regularization"
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.3_-dh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.4_-dh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_-dh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.4_-dh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.3_-dh")

# title = "Variational Graph Autoencoder + Spatial-aware GRU"
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.4_v-h")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.1_v-h")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.0_v-h")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.0_v-h")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_v-h")

# title = "Variational Graph Autoencoder + Spatial-aware GRU + Adverserial Regularization"
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold11_16.4_vdh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_vdh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold31_16.4_vdh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold41_16.1_vdh")
# savePATHS.append(r"/saved_models/FINAL/VGRNN_softmax_adv_fold51_16.1_vdh")

for savePATH in savePATHS:

	outer_loop = list(savePATH.split('_')[-3])[-2]
	train_graphs, test_graphs, val_graphs = load_data(outer_loop, inner_loop)

	dataset = myDataset(np.concatenate([train_graphs, val_graphs, test_graphs], axis=0))
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

	z_Samples.append(z_Sample)
	zh_Samples.append(zh_Sample)
	Readouts.append(Readout)
	Targets.append(Target)
	testIdx.append(test_idx)

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

saveTo = './saved_figures/'  
os.makedirs(saveTo, exist_ok=True)
plt.savefig(saveTo+title+'_Z'+'_'+init+'.jpg', dpi=400, bbox_inches='tight', pad_inches=0)
plt.show()
