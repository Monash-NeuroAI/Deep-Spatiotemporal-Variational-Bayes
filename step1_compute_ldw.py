#%%
from sklearn.covariance import LedoitWolf
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
import math

#%%
# Load data
def load_data():
    asd_data = np.load('./data/2_power_asd.npy', allow_pickle=True)
    td_data = np.load('./data/2_power_td.npy',allow_pickle=True)
    
    # Restructure data
    data = np.concatenate((asd_data,td_data))
    labels = np.concatenate(([np.ones(len(asd_data)), np.zeros(len(td_data))])).astype(int) # 1 : ASD, 0: TD
    
    # Check for missing ROIs data
    count1 = 0; count2 = 0
    nROIs = 264
    
    to_remove = []
    for i,x in enumerate(data):
        if x.shape[1]==nROIs:
            results = np.all((x == 0), axis=0)#ROI columns of all zeros
            if np.any(results):
                # idx = np.where(results)[0]
                to_remove.append(i)
                print('Data of subject {} is removed due to missing clumn ROI/s observations'.format(i))
                count1 +=1
        else:
            results =  np.all((x == 0), axis=1)#ROI rows of all zeros
            if np.any(results):
                # idx = np.where(results)[0]
                to_remove.append(i)
                print('Data of subject {} is removed due to missing row ROI/s observations'.format(i))
                count2 +=1
    if to_remove:
        data = np.delete(data, to_remove, 0)
        labels = np.delete(labels, to_remove, 0)
    return data, labels

# Compute the correlation & do thresholding
## Function for Converting Covariance to Correlation
def cov2corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

## Function for Thresholding
def threshold_proportional(W, p, copy=True):
    '''
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights, and
    all weights on the main diagonal (self-self connections) are set to 0.
    If copy is not set, this function will *modify W in place.*
    Inputs: W,      weighted or binary connectivity matrix
            p,      proportion of weights to preserve
                        range:  p=1 (all weights preserved) to
                                p=0 (no weights preserved)
            copy,    copy W to avoid side effects, defaults to True
    Output: W,        thresholded connectivity matrix
    Note: The proportion of elements set to 0 is a fraction of all elements in the 
    matrix, whether or not they are already 0.
    '''
    assert p < 1 or p > 0
    if copy:
        W = W.copy()
    n = len(W)                        # number of nodes
    np.fill_diagonal(W, 0)            # clear diagonal
    if np.all(W == W.T):                # if symmetric matrix
        W[np.tril_indices(n)] = 0        # ensure symmetry is preserved
        ud = 2                        # halve number of removed links
    else:
        ud = 1
    ind = np.where(W)                    # find all links
    I = np.argsort(W[ind])[::-1]        # sort indices by magnitude
    # number of links to be preserved
    en = round((n * n - n) * p / ud)
    W[(ind[0][I][en:], ind[1][I][en:])] = 0    # apply threshold
    if ud == 2:                        # if symmetric matrix
        W[:, :] = W + W.T                        # reconstruct symmetry
    
    W[W>0.9999] = 1                          # make sure the highest correlation coeff is 1
    return W

def extract_ldw_corr(data,wSize,shift):

    # Ledoit-Wolf optimal shrinkage coefficient estimate
    nSub = len(data)
    nROI = data[0].shape[1]
    #tpLen = min([item.shape[0] for item in data])
    tpLen = [item.shape[0] for item in data]
    
    # wSize = 40
    # shift = 10
    # overlap = wSize - shift
    # nWin = int((tpLen-overlap)/(wSize-overlap))
    # tpLen = min([item.shape[0] for item in data])
    # nWin = math.ceil(tpLen/wSize)
    
    overlap = wSize - shift
    nWin = [int((l-overlap)/(wSize-overlap)) for l in tpLen]
    
    # node_feats is X matrix of the Graph (ROIs, Time)
    # feat_len = min([item.shape[0] for item in data])
    # node_feats = np.zeros((nSub,nWin,nROI,nROI),dtype=float)
    # LDW_adj_mat  = np.zeros((nSub,nWin,nROI,nROI),dtype=float)
    # sub_labels = np.zeros((nSub,nWin),dtype=object)
    
    node_feats = [] # s,w,r,r
    LDW_adj_mat = []
    #sub_labels = np.zeros((nSub,nWin),dtype=object)

    # count = 0
    for sub in tqdm(range(len(data))):    # For each subject
        corr_mat = []
        adj_mat = []
        
        for wi in range(nWin[sub]):
            st = wi * (wSize - overlap)
            en = st + wSize
            w_data = data[sub][st:en,:]
            
        # print('Subject #{}'.format(i+1))
            lw = LedoitWolf(assume_centered=False)
            cov = lw.fit(w_data.squeeze())
            a = cov.covariance_
            corr_neg = cov2corr(a)
            corr = np.abs(corr_neg)
            # corr = np.abs(corr)
            corr_mat.append(corr_neg)

        # apply proportional thresholding
            th_corr = threshold_proportional(corr, 0.40)# keep top k% coeffs

        # fill daigonal with ones to avoid zero-degree nodes
            np.fill_diagonal(th_corr,1)
            adj_mat.append(th_corr)
            # count += 1

            #sub_labels[sub,wi] = 's%d_w%d'%(sub,wi)
        
        node_feats.append(corr_mat)
        LDW_adj_mat.append(adj_mat)

        # node_feats[i,...] = data[i][:feat_len,:].T
        assert np.all(np.logical_not(np.all((th_corr == 0), axis=1))), 'adjacency matrix contains rows of all zeros'
        assert np.all(np.logical_not(np.all((th_corr == 0), axis=0))), 'adjacency matrix contains columns of all zeros'
        assert np.all(th_corr>=0), 'adjacency matrix contains negative values'
        #w_labels = np.concatenate(([np.ones(70*nWin), np.zeros(74*nWin)])).astype(int)
        
    return node_feats, LDW_adj_mat, nWin

#%%
#def main():
import pickle
import dill
import os
# np.save("LDW_adj_mat.npy", LDW_adj_mat)
# np.save("SC_adj_mat.npy", SC_adj_mat)

data, labels = load_data()
data = [np.array(item) for item in data]

# import scipy.io as sio
# sio.savemat('abide_data.mat',{
#     'data':data,
#     'labels':labels})
# for fold_i in range(1,6):
# for iROI in range(1,4):
# print('Processing Fold_{} ....'.format(fold_i))
# iROI_path = '../irois/fold'+str(fold_i) +'/subset'+str(iROI)+'.npy'
# iROI_idx =  np.load(iROI_path, allow_pickle=True)

#iROIs_data = extract_iROIS(data,iROI_idx)

# Sliding window:
wSize = 20  # Change this windowSize
shift = 10  # Change this 

node_feats, adj_mats, nWin = extract_ldw_corr(data,wSize,shift)
#node_feats, adj_mats, sub_labels, nWin = extract_ldw_corr(iROIs_data,wSize,shift)

LDW_data = {}
LDW_data['adj_mat'] = adj_mats
LDW_data['node_feat'] = node_feats
LDW_data['labels'] = labels
#LDW_data['sub_labels'] = sub_labels

win_info = {}
win_info['wSize'] = wSize
win_info['shift'] = shift
win_info['nWin'] = nWin

saveTo = './ldw_data/'
if not os.path.exists(saveTo):
     os.makedirs(saveTo)
     
with open(saveTo+'LDW_abide_data.pkl', 'wb') as f:
    pickle.dump(LDW_data,f,protocol=4)
    
with open(saveTo+'win_info.pkl', 'wb') as f:
    pickle.dump(win_info,f,protocol=4)
# %%
