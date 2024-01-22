from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from model import *
from train import *

# Data loading and DataLoader setup
outer_loop = 1
inner_loop = 1
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

# Initialize training parameters
savePATH = './saved_models/FINAL/VGRNN_softmax_adv_fold' + str(outer_loop) + str(inner_loop) + '_16.10_v-h'
loadPATH = savePATH

model_params = {
    'num_nodes': 264, 'num_classes': 2, 'x_dim': 264, 'y_dim': 2,
    'z_hidden_dim': 32, 'z_dim': 16, 'z_phi_dim': 8,
    'x_phi_dim': 64, 'rnn_dim': 16, 'y_hidden_dim': [32],
    'x_hidden_dim': 64, 'layer_dims': []
}

lr_annealType = 'ReduceLROnPlateau'
lr_annealType = [lr_annealType, lr_annealType]

setting = {
    'rngPATH': './saved_models/FINAL/VGRNN_softmax_adv_fold21_16.0_vdh',
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

# Acquire model, optimizer, and scheduler
model, optimizers, schedulers, epochStart, train_losses, val_losses, test_losses = loadCheckpoint(setting, loadPATH, savePATH)
print(model)
print(model_params)
for optimizer in optimizers:
    print(optimizer)
for scheduler in schedulers:
    print(scheduler)
print(setting['rngPATH'])
print(savePATH)

# Train the model
model, train_losses, val_losses, test_losses = train(
    model, optimizers, schedulers, setting, savePATH,
    train_losses, val_losses, test_losses,
    train_loader, val_loader, test_loader,
    epochStart=epochStart, numEpochs=500,
    gradThreshold=1, gradientClip=True,
    verboseFreq=1, verbose=True, valFreq=1,
    validation=False, testing=True,
    earlyStopPatience=500, earlyStop=True)

# Analysis and Visualization
loadPATH = savePATH
checkpoint = torch.load(savePATH)
setting = checkpoint['training_setting']

plt.plot(torch.stack(checkpoint['train_losses']['z_KLD'], dim=0).cpu().numpy())
plt.plot(torch.stack(checkpoint['test_losses']['z_KLD'], dim=0).cpu().numpy())
plt.legend(['train', 'test'])
plt.show()

# Various plots for losses and accuracy
# Similar plotting steps are repeated for different metrics

# TSNE visualization
plt.style.use('default')
plt.style.use('seaborn-darkgrid')
init = 'default'
tsne = TSNE(n_components=2)

saveTo = './saved_figures/'
os.makedirs(saveTo, exist_ok=True)
plt.savefig(saveTo + title + '_Z' + '_' + init + '.jpg', dpi=
