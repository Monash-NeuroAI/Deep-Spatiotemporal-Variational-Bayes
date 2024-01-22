from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load data
def load_data():
    asd_data = np.load('./data/power_asd.npy', allow_pickle=True)
    td_data = np.load('./data/power_td.npy',allow_pickle=True)
    data = np.concatenate((asd_data, td_data))
    labels = np.concatenate(([np.ones(len(asd_data)), np.zeros(len(td_data))])).astype(int)

    nROIs = 264
    to_remove = []
    for i, x in enumerate(data):
        if x.shape[1] == nROIs:
            results = np.all((x == 0), axis=0)
            if np.any(results):
                to_remove.append(i)
        else:
            results = np.all((x == 0), axis=1)
            if np.any(results):
                to_remove.append(i)
    
    if to_remove:
        data = np.delete(data, to_remove, 0)
        labels = np.delete(labels, to_remove, 0)
    return data, labels

# Converting Covariance to Correlation
def cov2corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# Thresholding
def threshold_proportional(W, p, copy=True):
    assert p < 1 or p > 0
    if copy:
        W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)
    ud = 2 if np.all(W == W.T) else 1
    ind = np.where(W)
    I = np.argsort(W[ind])[::-1]
    en = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0
    if ud == 2:
        W[:, :] = W + W.T
    
    W[W > 0.9999] = 1
    return W

def extract_ldw_corr(data, wSize, shift):
    nSub = len(data)
    nROI = data[0].shape[1]
    tpLen = [item.shape[0] for item in data]

    overlap = wSize - shift
    nWin = [int((l-overlap)/(wSize-overlap)) for l in tpLen]

    node_feats = []
    LDW_adj_mat = []

    for sub in tqdm(range(len(data))):
        corr_mat = []
        adj_mat = []
        
        for wi in range(nWin[sub]):
            st = wi * (wSize - overlap)
            en = st + wSize
            w_data = data[sub][st:en,:]

            lw = LedoitWolf(assume_centered=False)
            cov = lw.fit(w_data.squeeze())
            a = cov.covariance_
            corr_neg = cov2corr(a)
            corr = np.abs(corr_neg)
            corr_mat.append(corr_neg)

            th_corr = threshold_proportional(corr, 0.40)
            np.fill_diagonal(th_corr, 1)
            adj_mat.append(th_corr)
        
        node_feats.append(corr_mat)
        LDW_adj_mat.append(adj_mat)
        
    return node_feats, LDW_adj_mat, nWin

# Main processing
import pickle
import os

data, labels = load_data()
data = [np.array(item) for item in data]

wSize = 20
shift = 10

node_feats, adj_mats, nWin = extract_ldw_corr(data, wSize, shift)

LDW_data = {
    'adj_mat': adj_mats,
    'node_feat': node_feats,
    'labels': labels
}

win_info = {
    'wSize': wSize,
    'shift': shift,
    'nWin': nWin
}

saveTo = './ldw_data/'
if not os.path.exists(saveTo):
    os.makedirs(saveTo)

with open(saveTo+'LDW_abide_data.pkl', 'wb') as f:
    pickle.dump(LDW_data, f, protocol=4)
    
with open(saveTo+'win_info.pkl', 'wb') as f:
    pickle.dump(win_info, f, protocol=4)
