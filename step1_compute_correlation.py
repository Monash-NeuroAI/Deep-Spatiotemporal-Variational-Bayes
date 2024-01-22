from sklearn.covariance import LedoitWolf
import numpy as np
from tqdm import tqdm
import os
import pickle

# Load data
def load_data():
    asd_data = np.load('data/power_asd.npy', allow_pickle=True)
    td_data = np.load('data/power_td.npy', allow_pickle=True)
    data = np.concatenate((asd_data, td_data))
    labels = np.concatenate(([np.ones(len(asd_data)), np.zeros(len(td_data))])).astype(int)
    to_remove = []
    nROIs = 264
    for i, x in enumerate(data):
        if x.shape[1] == nROIs:
            if np.any(np.all((x == 0), axis=0)):
                to_remove.append(i)
        else:
            if np.any(np.all((x == 0), axis=1)):
                to_remove.append(i)
    if to_remove:
        data = np.delete(data, to_remove, 0)
        labels = np.delete(labels, to_remove, 0)
    return data, labels

# Convert Covariance to Correlation
def cov2corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# Thresholding Function
def threshold_proportional(W, p, copy=True):
    assert p < 1 or p > 0
    if copy:
        W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)
    ud = 2 if np.all(W == W.T) else 1
    W[np.tril_indices(n)] = 0 if ud == 2 else W
    ind = np.where(W)
    I = np.argsort(W[ind])[::-1]
    en = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0
    if ud == 2:
        W[:, :] = W + W.T
    W[W > 0.9999] = 1
    return W

# Extract LDW Correlation
def extract_ldw_corr(data, wSize, shift):
    nSub = len(data)
    nROI = data[0].shape[1]
    tpLen = [item.shape[0] for item in data]
    overlap = wSize - shift
    nWin = [int((l-overlap)/(wSize-overlap)) for l in tpLen]
    node_feats, LDW_adj_mat = [], []

    for sub in tqdm(range(nSub)):
        corr_mat, adj_mat = [], []
        for wi in range(nWin[sub]):
            st, en = wi * (wSize - overlap), wi * (wSize - overlap) + wSize
            w_data = data[sub][st:en, :]
            lw = LedoitWolf(assume_centered=False)
            cov = lw.fit(w_data.squeeze())
            corr_neg = cov2corr(cov.covariance_)
            corr = np.abs(corr_neg)
            corr_mat.append(corr_neg)
            th_corr = threshold_proportional(corr, 0.40)
            np.fill_diagonal(th_corr, 1)
            adj_mat.append(th_corr)
        node_feats.append(corr_mat)
        LDW_adj_mat.append(adj_mat)
    return node_feats, LDW_adj_mat, nWin

# Main processing function
def process_data():
    data, labels = load_data()
    wSize, shift = 20, 10
    node_feats, adj_mats, nWin = extract_ldw_corr(data, wSize, shift)
    LDW_data = {'adj_mat': adj_mats, 'node_feat': node_feats, 'labels': labels}
    win_info = {'wSize': wSize, 'shift': shift, 'nWin': nWin}
    saveTo = 'ldw_data/'
    if not os.path.exists(saveTo):
        os.makedirs(saveTo)
    with open(os.path.join(saveTo, 'LDW_abide_data.pkl'), 'wb') as f:
        pickle.dump(LDW_data, f, protocol=4)
    with open(os.path.join(saveTo, 'win_info.pkl'), 'wb') as f:
        pickle.dump(win_info, f, protocol=4)

# Uncomment to run the main function
# process_data
