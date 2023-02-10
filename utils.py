import os
import random

import torch
import dgl

import scipy.sparse as sp
import numpy as np
from tensorflow.keras.utils import to_categorical


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask
def apply_feature_mask(features, mask):
    features[mask] = float('nan')

def edge_delete(prob_del, adj, enforce_connected=False):
    rnd = np.random.RandomState(1234)
    adj= adj.toarray()
    del_adj = np.array(adj, dtype=np.float32)
    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()
    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
    del_adj= sp.csr_matrix(del_adj)

    return del_adj


def load_data(dataset, repeat, device, rate):
    path = './data/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
    train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
    val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    mask = feature_mask(features, rate)
    apply_feature_mask(features, mask)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    label = torch.LongTensor(np.array(l)).to(device)

    label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = edge_delete(rate, sadj)

    # ppr_input:A+I
    ttadj = sadj + sp.eye(sadj.shape[0])
    ttadj = torch.FloatTensor(ttadj.todense()).to(device)
    # A
    tadj = torch.FloatTensor(sadj.todense()).to(device)
    # stu_input
    sadj = normalize_sparse(sadj + sp.eye(sadj.shape[0]))
    nsadj = torch.FloatTensor(np.array(sadj.todense())).to(device)


    return ttadj, tadj, nsadj, features, label, label_oneHot, idx_train, idx_val, idx_test
