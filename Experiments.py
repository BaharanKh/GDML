import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation, PCA
import pickle as pkl
import random
import sys
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import networkx as nx
from networkx.algorithms import community
from torch_geometric.datasets import Planetoid
import scipy.special as sp
import networkx as nx
from sklearn.linear_model import LinearRegression
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import argparse
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.loader import NeighborLoader
from numpy.linalg import inv


def dgp0(X, E, gamma=1.0):
    X = X.to_numpy()
    piZ = sp.expit(np.mean(X, axis=1) + np.random.normal(0, 1, size=X.shape[0]))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + 10 + np.random.normal(0, 1, size=X.shape[0])
    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true


def dgp1(X, E, gamma=1.0):
    X = X.to_numpy()
    piZ = np.array([0.5] * len(X))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + 10 + np.random.normal(0, 1, size=X.shape[0])
    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true


def dgp2(X, E, gamma=1.0):
    X = X.to_numpy()
    piZ = np.array([0.5] * len(X))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                       axis=1) + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                       axis=1) + 10 + np.random.normal(0, 1,
                                                                                                       size=X.shape[0])
    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true


def dgp3(X, E, gamma=1.0):
    X = X.to_numpy()
    piZ = sp.expit(np.mean(X, axis=1) + np.random.normal(0, 1, size=X.shape[0]))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                       axis=1) + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                       axis=1) + 10 + np.random.normal(0, 1,
                                                                                                       size=X.shape[0])
    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true


def dgp4(X, E, gamma=1.0):
    X = X.to_numpy()
    piZ = sp.expit(
        np.mean(X, axis=1) + gamma * np.mean(np.matmul(E, X), axis=1) + np.random.normal(0, 1, size=X.shape[0]))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                       axis=1) + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + 10 + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                            axis=1) + np.random.normal(0, 1,
                                                                                                       size=X.shape[0])
    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true


# def dgp5(E, gamma=1.0):
#     X = np.random.normal(size=(E.shape[0], 1))
#     piZ = sp.expit(np.mean(X, axis=1) + gamma * np.mean(np.matmul(E, X), axis=1) + np.random.normal(0, 1, size=X.shape[0]))
#     Z = np.random.binomial(1, piZ)
#     Y0 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)), axis=1) + np.random.normal(0, 1, size=X.shape[0])
#     Y1 = np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1) + 10 + np.mean(np.matmul(E, Z.reshape(-1, 1)), axis=1) + np.random.normal(0, 1, size=X.shape[0])
#     Y = Y1 * Z + Y0 * (1 - Z)
#     df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
#                       index=['u%d' % (i) for i in range(X.shape[0])])
#     df['Y'] = Y
#     df['Z'] = Z
#     network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
#                            index=['u%d' % (i) for i in range(X.shape[0])])
#     df_true = pd.DataFrame()
#     df_true['Y1'] = Y1
#     df_true['Y0'] = Y0
#     df_true['piZ'] = piZ
#     return df, network, df_true, X

def dgp5(E, gt=10, beta=5, gamma=1.0):
    X = np.random.normal(size=(E.shape[0], 1))

    piZ = sp.expit((np.mean(X, axis=1) + gamma * np.mean(np.matmul(E, X),
                                                         axis=1)) / 10)  # + np.random.normal(0, 1, size=X.shape[0]))
    Z = np.random.binomial(1, piZ)

    deg = E.sum(axis=1)
    # peer_effect = (np.matmul(E, Z) > deg/2).astype(int)
    # peer_effect = np.matmul(E, Z) / deg
    peer_effect = np.matmul(E, Z)

    Y = Z * gt + peer_effect * beta + np.sum(X, axis=1) + np.sum(np.matmul(E, X), axis=1)

    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()

    df_true['piZ'] = piZ
    return df, network, df_true, X


def dgp6(E, gt=10, gamma=1.0):
    X = np.random.normal(loc=10, size=(E.shape[0], 1))
    piZ = sp.expit((np.sum(X, axis=1) + gamma * np.max(np.multiply(E, X),
                                                       axis=1)) / 100)  # + np.random.normal(0, 1, size=X.shape[0]))
    Z = np.random.binomial(1, piZ)
    Y0 = np.sum(X, axis=1) + np.max(np.multiply(E, X), axis=1) + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                         axis=1)  # + np.random.normal(0, 1, size=X.shape[0])
    Y1 = np.sum(X, axis=1) + np.max(np.multiply(E, X), axis=1) + gt + np.mean(np.matmul(E, Z.reshape(-1, 1)),
                                                                              axis=1)  # + np.random.normal(0, 1, size=X.shape[0])

    # Y0 = (Y0-Y0.mean())/Y0.std()
    # Y1 = Y0 + 10

    Y = Y1 * Z + Y0 * (1 - Z)
    df = pd.DataFrame(X, columns=['X%d' % (i) for i in range(X.shape[1])],
                      index=['u%d' % (i) for i in range(X.shape[0])])
    df['Y'] = Y
    df['Z'] = Z
    network = pd.DataFrame(E, columns=['u%d' % (i) for i in range(X.shape[0])],
                           index=['u%d' % (i) for i in range(X.shape[0])])
    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piZ'] = piZ
    return df, network, df_true, X


class CustomDataset(InMemoryDataset):
    def __init__(self, A, x, y, input_feature_type, target_feature_type, x1, x2, transform=None):
        super(CustomDataset, self).__init__('.', transform, None, None)
        edge_list = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)

        if input_feature_type == 'discrete' and type(x).__module__ == np.__name__:
            x = torch.tensor(x, dtype=torch.long)
        elif input_feature_type == 'continuous' and type(x).__module__ == np.__name__:
            x = torch.tensor(x, dtype=torch.float)

        if target_feature_type == 'discrete' and type(y).__module__ == np.__name__:
            y = torch.tensor(y, dtype=torch.long).flatten()
        elif target_feature_type == 'continuous' and type(y).__module__ == np.__name__:
            y = torch.tensor(y, dtype=torch.float).flatten()

        # Creating train and test masks
        n_nodes = len(x)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[x1] = True
        test_mask[x2] = True

        self.data = Data(x=x, y=y, edge_index=edge_list, train_mask=train_mask, test_mask=test_mask)

        if target_feature_type == 'discrete':
            self.data.num_classes = len(y.unique())

        self.data.num_node_features = x.shape[1]

        self.data.input_feature_type = input_feature_type
        self.data.target_feature_type = target_feature_type

        self.data, self.slices = self.collate([self.data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def subgraph(self, mask):
        return self.data.subgraph(mask)

    @classmethod
    def from_data(cls, data, input_feature_type, target_feature_type):
        A = torch.zeros((data.x.shape[0], data.x.shape[0]), dtype=torch.long)
        A[data.edge_index[0], data.edge_index[1]] = 1
        x = data.x
        y = data.y
        x1 = data.train_mask.nonzero(as_tuple=True)[0]
        x2 = data.test_mask.nonzero(as_tuple=True)[0]

        return cls(A, x, y, input_feature_type, target_feature_type, x1, x2)


def covariate_transform(X, dimension):
    lda = LatentDirichletAllocation(n_components=dimension)
    X = lda.fit_transform(X)
    return X


def covariate_transform_pca(X, dimension):
    pca = PCA(n_components=dimension)
    X = pca.fit_transform(X)
    return X


def cora_process(args):
    raw_data = pd.read_csv('../data/Cora/cora.content', sep='\t', header=None)
    num = raw_data.shape[0]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b, a)
    map = dict(c)

    features = raw_data.iloc[:, 1:-1]
    labels = pd.get_dummies(raw_data[1434])
    raw_data_cites = pd.read_csv('../data/Cora/cora.cites', sep='\t', header=None)

    matrix = np.zeros((num, num))

    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = map[i]
        y = map[j]
        matrix[x][y] = matrix[y][x] = 1

    arr = np.array(features)

    if args.transform == 'pca':
        arr = covariate_transform_pca(arr, 10)
    elif args.transform == 'lda':
        arr = covariate_transform(arr, 10)

    return arr, matrix


def citeseer_process(args):
    raw_data = pd.read_csv('../data/Citeseer/citeseer.content', sep='\t', header=None)

    raw_data_cites = pd.read_csv('../data/Citeseer/citeseer.cites', sep='\t', header=None)

    ct_idx = list(raw_data.index)
    paper_id = list(raw_data.iloc[:, 0])
    paper_id = [str(i) for i in paper_id]
    mp = dict(zip(paper_id, ct_idx))

    label = raw_data.iloc[:, -1]
    label = pd.get_dummies(label)
    label.shape

    features = raw_data.iloc[:, 1:-1]

    mlen = raw_data.shape[0]
    matrix = np.zeros((mlen, mlen))

    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        if str(i) in mp.keys() and str(j) in mp.keys():
            x = mp[str(i)]
            y = mp[str(j)]
            matrix[x][y] = matrix[y][x] = 1

    arr = np.array(features)

    if args.transform == 'pca':
        arr = covariate_transform_pca(arr, 10)
    elif args.transform == 'lda':
        arr = covariate_transform(arr, 10)

    return arr, matrix


def pubmed_process(args):
    # with open('../data/Pubmed/Pubmed-Diabetes.NODE.paper.tab') as f:
    #     for line in f:
    #         first = line.strip().split('\t')
    #         break

    #     # next(f)
    #     features = []
    #     nodes = []
    #     for line in f:
    #         cols = line.strip().split('\t')
    #         node_id = cols[0]
    #         feature_vec = np.array([[x.split('=')[0], x.split('=')[1]] for x in cols[1:]])
    #         features.append(feature_vec)
    #         nodes.append(node_id)

    # names = []
    # for i in first[1:-1]:
    #     names.append(i[8:-4])

    # id = []
    # with open('../data/Pubmed/Pubmed-Diabetes.NODE.paper.tab') as f:
    #     next(f)
    #     for line in f:
    #         cols = line.strip().split('\t')
    #         id.append(cols[0])
    # len(id)

    # df = pd.DataFrame(columns=names)

    # features_new = []
    # for i in features:
    #     features_new.append(i[1:-1])

    # from tqdm import tqdm
    # for feature in tqdm(features_new):
    #     s = pd.Series(np.zeros(len(names)), index=names)
    #     for i in feature:
    #         if i[0] in names:
    #             s[i[0]] = i[1]
    #     df = df.append(s, ignore_index=True)

    # df.index = pd.Index(id)

    # arr = np.array(df)

    # if args.transform == 'pca':
    #     arr = covariate_transform_pca(arr, 10)
    # elif args.transform == 'lda':
    #     arr = covariate_transform(arr, 10)

    # matrix = np.zeros(2).reshape(1, 2)
    # from tqdm import tqdm
    # with open('../data/Pubmed/Pubmed-Diabetes.DIRECTED.cites.tab') as f:
    #     for line in tqdm(f):
    #         cols = line.strip().split('\t')
    #         matrix = np.append(matrix, np.array([[cols[1][6:], cols[3][6:]]]), axis=0)

    # matrix = matrix[1:]

    # mlen = arr.shape[0]

    # mat = np.zeros((mlen, mlen))
    # mp = {}

    # for i in range(len(df)):
    #     mp[df.index[i]] = i

    # for i,j in zip(matrix[:,0], matrix[:,1]):
    #     if str(i) in mp.keys() and str(j) in mp.keys():
    #         x = mp[str(i)]
    #         y = mp[str(j)]
    #         mat[x][y] = mat[y][x] = 1

    # matrix, features = mat, arr

    # pkl.dump(matrix, open('../data/Pubmed/matrix.pkl', 'wb'))
    # pkl.dump(features, open('../data/Pubmed/features.pkl', 'wb'))

    matrix = pkl.load(open('../data/Pubmed/matrix.pkl', 'rb'))
    features = pkl.load(open('../data/Pubmed/features.pkl', 'rb'))

    return features, matrix


def BC_process(args):
    nodes = pd.read_csv('../data/BC/nodes.csv', names=['id'])
    num = list(nodes['id'])[-1]
    df = pd.read_csv('../data/BC/edges.csv', names=['v1', 'v2'])
    edges = list(zip(df['v1'], df['v2']))
    edges.insert(0, (1, 2))
    matrix = np.zeros((num, num), dtype=int)
    for e in edges:
        matrix[e[0] - 1][e[1] - 1] = 1
        matrix[e[1] - 1][e[0] - 1] = 1
    return matrix


def sbm_process(args):
    n_nodes = 3000
    n_comps = 200
    sizes = list(np.ones(n_comps, dtype=int) * (n_nodes // n_comps))
    probs = np.ones((n_comps, n_comps)) * 0.0001
    for i in range(n_comps):
        probs[i][i] = 0.75

    G = nx.stochastic_block_model(sizes, probs, seed=0)
    matrix = nx.adjacency_matrix(G).toarray()
    print('Number of edges in SBM graph: ', matrix.sum() / 2)
    return matrix


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def find_focal_set(matrix):
    maximal_set = set()
    idx = list(range(len(matrix)))
    remained_nodes = set(idx)
    print(matrix.shape)
    while remained_nodes:
        if len(remained_nodes) % 5 == 0:
            print(len(remained_nodes))
        remove_nodes = set()

        min_degree_idx = np.argmin(matrix[list(remained_nodes)][:, list(remained_nodes)].sum(axis=1))
        # min_degree_idx = random.choice(list(remained_nodes))

        maximal_set.add(idx[min_degree_idx])
        remove_nodes.add(idx[min_degree_idx])

        neighbors = list(np.where(matrix[idx[min_degree_idx]] == 1)[0])
        remove_nodes.update(neighbors)

        for n in neighbors:
            n_neighbors = np.where(matrix[n] == 1)[0]
            remove_nodes.update(n_neighbors)

        rem = sorted(list(remove_nodes.intersection(remained_nodes)), reverse=True)
        for r in rem:
            idx.remove(r)
        remained_nodes = remained_nodes - remove_nodes

    return maximal_set


class GCN(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2):
        super().__init__()

        self.feature_type = feature_type
        if feature_type == 'discrete':
            self.conv1 = GCNConv(num_node_features, num_classes)

        elif feature_type == 'continuous':
            self.conv1 = GCNConv(num_node_features, 8)
            self.fc1 = torch.nn.Linear(8, 1)

    #             self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.feature_type == 'discrete':
            return F.log_softmax(x, dim=1), x
        else:
            x = self.fc1(x)
            #             x = self.fc2(x)
            return x

    def fit(self, dataset, optimizer, feature_type, n_epoch=100):
        losses = []
        self.train()
        for epoch in range(1, n_epoch + 1):
            #         model.train()
            optimizer.zero_grad()
            if feature_type == 'discrete':
                #             m = nn.Sigmoid()
                #             loss = nn.BCELoss()
                #             print(dataset.y[dataset.train_mask].type())
                #             print(torch.argmax(model(dataset)[dataset.train_mask], dim=1).type())
                #             loss = loss(m(torch.argmax(model(dataset)[dataset.train_mask], dim=1)), dataset.y[dataset.train_mask].float())
                loss = F.nll_loss(self(dataset.x, dataset.edge_index)[0][dataset.train_mask],
                                  dataset.y[dataset.train_mask])

            #             loss = F.cross_entropy(model(dataset)[dataset.train_mask], dataset.y[dataset.train_mask])

            #             print(dataset.y[dataset.train_mask].shape)
            #             print(torch.argmax(model(dataset)[dataset.train_mask], dim=1).type())
            #             loss = F.mse_loss(model(dataset)[dataset.train_mask], dataset.y[dataset.train_mask])

            elif feature_type == 'continuous':
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(self(dataset.x, dataset.edge_index).flatten()[dataset.train_mask],
                                dataset.y[dataset.train_mask])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch ', epoch, ' done!')

        if feature_type == 'continuous':
            plt.plot(list(range(1, n_epoch + 1)), losses)

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)
            correct = (pred.argmax(dim=1)[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            print('MSE loss for test data: ', output)
        return pred


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2):
        super().__init__()

        self.feature_type = feature_type
        if feature_type == 'discrete':
            self.sage1 = SAGEConv(num_node_features, num_classes, 'mean')

        elif feature_type == 'continuous':
            self.sage1 = SAGEConv(num_node_features, 8, 'mean')
            self.fc1 = torch.nn.Linear(8, 1)

    #             self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        if self.feature_type == 'discrete':
            return F.log_softmax(x, dim=1), x
        else:
            x = self.fc1(x)
            #             x = self.fc2(x)
            return x

    def fit(self, dataset, optimizer, feature_type, n_epoch=200):

        # Create batches with neighbor sampling
        train_loader = NeighborLoader(
            dataset,
            num_neighbors=[5, 10],
            batch_size=16,
            input_nodes=dataset.train_mask,
        )
        losses = []
        self.train()
        for epoch in range(1, n_epoch + 1):
            total_loss = 0
            acc = 0

            for batch in train_loader:
                optimizer.zero_grad()

                if feature_type == 'discrete':
                    out, _ = self(batch.x, batch.edge_index)
                    loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                    #                     loss = torch.nn.CrossEntropyLoss()(out[batch.train_mask], batch.y[batch.train_mask])
                    acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                    batch.y[batch.train_mask])

                elif feature_type == 'continuous':
                    out = self(batch.x, batch.edge_index)
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(torch.reshape(out[batch.train_mask], (-1,)), batch.y[batch.train_mask])

                total_loss += loss
                loss.backward()
                optimizer.step()

            # if (epoch % 50 == 0):
            #     if feature_type == 'discrete':
            #         print(
            #             f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f} 'f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}%')
            #
            #     else:
            #         print(f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f}')
            losses.append(total_loss.item())
        if feature_type == 'continuous':
            plt.plot(list(range(1, n_epoch + 1)), losses)

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)
            correct = (pred.argmax(dim=1)[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            print('MSE loss for test data: ', output)
        return pred


class GIN(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2, dim_h=32):
        super(GIN, self).__init__()
        self.feature_type = feature_type
        self.gin1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h, dim_h * 3)

        if feature_type == 'discrete':
            self.lin2 = Linear(dim_h * 3, num_classes)
        elif feature_type == 'continuous':
            self.lin2 = Linear(dim_h * 3, 1)

    def forward(self, x, edge_index):
        # Node embeddings
        h1 = self.gin1(x, edge_index)

        # Graph-level readout
        # h1 = global_add_pool(h1, batch)

        # Concatenate graph embeddings
        # h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h1)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        if self.feature_type == 'discrete':
            return F.log_softmax(h, dim=1), h
        else:
            return h

    def fit(self, dataset, optimizer, feature_type, n_epoch=300):

        # Create batches

        train_dataset = dataset.subgraph(dataset.train_mask)
        train_dataset = CustomDataset.from_data(train_dataset, dataset.input_feature_type, dataset.target_feature_type)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        self.train()
        for epoch in range(n_epoch + 1):
            total_loss = 0
            acc = 0

            for batch in train_loader:
                optimizer.zero_grad()

                if feature_type == 'discrete':
                    out = self(batch.x, batch.edge_index.T)
                    loss = F.nll_loss(out[0][batch.train_mask], batch.y[batch.train_mask])
                    acc += accuracy(out[0][batch.train_mask].argmax(dim=1),
                                    batch.y[batch.train_mask])

                elif feature_type == 'continuous':
                    out = self(batch.x, batch.edge_index.T)
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(torch.reshape(out[batch.train_mask], (-1,)), batch.y[batch.train_mask])

                total_loss += loss

                loss.backward()
                optimizer.step()

            # if(epoch % 50 == 0):
            #    if feature_type == 'discrete':
            #        print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} 'f'| Train Acc: {acc/len(train_loader)*100:>6.2f}%')
            #
            #    else:
            #       print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f}')

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)[0].argmax(dim=1)
            correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            # print('MSE loss for test data: ', output)
        return pred


def find_train_split(splits, ind):
    train = []
    for i, l in enumerate(splits):
        if i == ind:
            continue
        train += l

    return train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--transform', type=str, default='pca')
    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--gt', type=int, default=10)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--K', type=int, default=3)
    args = parser.parse_args()

    data_use = args.dataset
    print('Start Processing Dataset......\n')

    if data_use == 'cora':
        features, matrix = cora_process(args)

    if data_use == 'pubmed':
        features, matrix = pubmed_process(args)

    if data_use == 'SBM':
        matrix = sbm_process(args)

    # maximal_set = find_focal_set(matrix)
    maximal_set = set(list(range(len(matrix))))
    # with open(f'focal_{data_use}.txt','w') as f:
    #     f.write(str(maximal_set))  # set of numbers & a tuple
    # print('focal set dumped')
    # f = open(f"../data/focal_{data_use}.txt", "r")
    # s = f.read()[1:-1].split(', ')
    # maximal_set = set([int(node) for node in s])

    print('focal set size: ', len(maximal_set))
    print('edges: ', matrix.sum() // 2)

    E, gamma = matrix, args.gamma
    ####
    trials = args.trials
    # trials = 1
    ####
    gt = args.gt
    beta = args.beta
    K = args.K

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    np.random.seed(0)

    ADEs = []
    APEs = []
    runtime = []
    variances = []
    n_covariates = 1
    n_coverage_theta, n_coverage_alpha = 0, 0
    avg_std = 0
    for tr in range(trials):
        start = time.time()

        print('run ', tr)

        dpg_n = 5
        ####
        data, A, df_true, X = dgp5(matrix)

        data.rename(columns={'Z': 'T'}, inplace=True)

        # A = A.to_numpy()
        A = matrix
        ####
        G = nx.from_numpy_matrix(A)

        theta = 0
        alpha = 0

        J0 = np.zeros((2, 2))
        var = np.zeros((2, 2))
        inds = list(maximal_set)
        print('len of each fold: ', len(inds) // K)

        random.shuffle(inds)
        splits = [inds[len(inds) * i // K:len(inds) * (i + 1) // K] for i in range(K)]
        print(A.shape, E.shape, matrix.shape)
        for i in range(K):
            psi = np.zeros((2, 1))

            x1 = np.array(find_train_split(splits, i))
            x2 = np.array(splits[i])

            x1.sort()
            x2.sort()

            treatment_dataset1 = \
            CustomDataset(A, data[data.columns[0:n_covariates]].to_numpy(), data['T'].to_numpy(), 'continuous',
                          'discrete', x1, x2)[0].to(device)
            # treatment_dataset2 = CustomDataset(A, data[data.columns[0:n_covariates]].to_numpy(), data['T'].to_numpy(), 'continuous', 'discrete', x2, x1)[0].to(device)
            print('datasets created!')

            ######GNN
            if args.model == 'GraphSage':
                treatment_model1 = GraphSAGE('discrete', treatment_dataset1.x.shape[1],
                                             len(torch.unique(treatment_dataset1.y))).to(device)
                # treatment_model2 = GraphSAGE('discrete', treatment_dataset2.x.shape[1], len(torch.unique(treatment_dataset2.y))).to(device)
            elif args.model == 'GCN':
                treatment_model1 = GCN('discrete', treatment_dataset1.x.shape[1],
                                       len(torch.unique(treatment_dataset1.y))).to(device)
                # treatment_model2 = GCN('discrete', treatment_dataset2.x.shape[1], len(torch.unique(treatment_dataset2.y))).to(device)
            elif args.model == 'GIN':
                treatment_model1 = GIN('discrete', treatment_dataset1.x.shape[1],
                                       len(torch.unique(treatment_dataset1.y))).to(device)
                # treatment_model2 = GIN('discrete', treatment_dataset2.x.shape[1], len(torch.unique(treatment_dataset2.y))).to(device)

            treatment_optimizer1 = torch.optim.Adam(treatment_model1.parameters(), lr=0.01, weight_decay=5e-4)
            # treatment_optimizer2 = torch.optim.Adam(treatment_model2.parameters(), lr=0.01, weight_decay=5e-4)
            treatment_model1.fit(treatment_dataset1, treatment_optimizer1, 'discrete')
            # treatment_model2.fit(treatment_dataset2, treatment_optimizer2, 'discrete')
            # t_hat1 = torch.exp(treatment_model2(treatment_dataset2.x, treatment_dataset2.edge_index)[0])[
            #              np.array(treatment_dataset2.test_mask)][:, 1].detach().numpy()
            t_hat = torch.exp(treatment_model1(treatment_dataset1.x, treatment_dataset1.edge_index)[0])[:,
                    1].detach().numpy()
            # print(accuracy(treatment_model1(treatment_dataset1.x, treatment_dataset1.edge_index)[0].argmax(dim=1), treatment_dataset1.y))
            res_T = data['T'] - t_hat
            # print('hereeeeee0')
            J0[0][0] += (-np.dot(res_T[np.array(treatment_dataset1.test_mask.cpu())],
                                 res_T[np.array(treatment_dataset1.test_mask.cpu())]))
            #
            # print('hereeeeee1')
            J0[0][1] += (-np.sum(np.multiply(np.matmul(np.transpose(res_T), np.transpose(E)), res_T)[
                                     np.array(treatment_dataset1.test_mask.cpu())]))
            #
            # print('hereeeeee2')
            J0[1][0] += (-np.sum(
                np.multiply(np.matmul(np.transpose(res_T), E), res_T)[np.array(treatment_dataset1.test_mask.cpu())]))
            #
            # print('hereeeeee3')
            J0[1][1] += (-np.sum(np.multiply(np.matmul(np.transpose(res_T), np.matmul(np.transpose(E), E)), res_T)[
                                     np.array(treatment_dataset1.test_mask.cpu())]))
            # print('hereeeeee4')
            J0 /= len(splits[i])


            outcome_dataset1 = \
            CustomDataset(A, data[data.columns[0:n_covariates]].to_numpy(), data['Y'].to_numpy(), 'continuous',
                          'continuous', x1, x2)[0].to(device)
            # outcome_dataset2 = CustomDataset(A, data[data.columns[0:n_covariates]].to_numpy(), data['Y'].to_numpy(), 'continuous', 'continuous', x2, x1)[0].to(device)

            if args.model == 'GraphSage':
                outcome_model1 = GraphSAGE('continuous', outcome_dataset1.x.shape[1]).to(device)
                # outcome_model2 = GraphSAGE('continuous', outcome_dataset2.x.shape[1]).to(device)
            elif args.model == 'GCN':
                outcome_model1 = GCN('continuous', outcome_dataset1.x.shape[1]).to(device)
                # outcome_model2 = GCN('continuous', outcome_dataset2.x.shape[1]).to(device)
            elif args.model == 'GIN':
                outcome_model1 = GIN('continuous', outcome_dataset1.x.shape[1]).to(device)
                # outcome_model2 = GIN('continuous', outcome_dataset2.x.shape[1]).to(device)

            outcome_optimizer1 = torch.optim.Adam(outcome_model1.parameters(), lr=0.01, weight_decay=5e-4)
            # outcome_optimizer2 = torch.optim.Adam(outcome_model2.parameters(), lr=0.01, weight_decay=5e-4)

            outcome_model1.fit(outcome_dataset1, outcome_optimizer1, 'continuous')

            y_hat = outcome_model1.test(outcome_dataset1, 'continuous').detach().numpy()
            res_Y = data['Y'] - y_hat
            print(E.shape, res_T.shape, data.shape)
            res_PE = np.matmul(E, res_T)

            reg = LinearRegression().fit(np.concatenate((res_T.to_numpy().reshape(-1, 1)[
                                                             np.array(treatment_dataset1.test_mask.cpu())],
                                                         res_PE.reshape(-1, 1)[
                                                             np.array(treatment_dataset1.test_mask.cpu())],
                                                         treatment_dataset1.x[
                                                             np.array(treatment_dataset1.test_mask.cpu())]), axis=1),
                                         res_Y[np.array(treatment_dataset1.test_mask)].to_numpy())

            theta += reg.coef_[0]
            alpha += reg.coef_[1]
            # print(res_T, res_Y)
            psi0 = np.multiply(res_Y - np.matmul((reg.coef_[0] + reg.coef_[1] * E), res_T), res_T)
            psi1 = np.multiply(np.matmul(np.transpose(res_Y - np.matmul((reg.coef_[0] + reg.coef_[1] * E), res_T)), E),
                               res_T)
            for j in x2:
                psi[0][0] = psi0.iloc[j]
                psi[1][0] = psi1.iloc[j]
                # print(psi, np.matmul(psi, np.transpose(psi)))
                var += np.matmul(psi, np.transpose(psi))
            var /= len(splits[i])

        print(theta / K, alpha / K)
        theta /= K
        alpha /= K
        J0 /= K
        var /= K
        var = np.matmul(np.matmul(inv(J0), var), np.transpose(inv(J0)))

        # print('var: ', var)

        CI_theta_start = theta - (var[0][0] ** 0.5) * 1.96 / (len(inds))
        CI_theta_end = theta + (var[0][0] ** 0.5) * 1.96 / (len(inds))
        
        CI_alpha_start = alpha - (var[1][1] ** 0.5) * 1.96 / (len(inds))
        CI_alpha_end = alpha + (var[1][1] ** 0.5) * 1.96 / (len(inds))

        print(CI_theta_start, CI_theta_end)
        print(CI_alpha_start, CI_alpha_end)
        avg_std += (var[0][0] ** 0.5) / (len(inds)) ** 0.5

        if CI_theta_start <= 10 and CI_theta_end >= 10:
            n_coverage_theta += 1
        
        if CI_alpha_start <= 5 and CI_alpha_end >= 5:
            n_coverage_alpha += 1

        variances.append(var)
        ADEs.append(theta)
        APEs.append(alpha)
        runtime.append(time.time() - start)

    # print('avg std: ', avg_std / trials)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # df = pd.DataFrame(list(zip(ADEs, APEs, variances)), columns=['ADE', 'AIE', 'VAR'])
    df = pd.DataFrame(list(zip(ADEs, APEs, variances, runtime)), columns=['ADE', 'AIE', 'VAR', 'runtime'])
    df.to_csv(f'../results/{args.dataset}_{timestr}.csv', index=False)
    print(n_coverage_theta / trials, n_coverage_alpha / trials)

    print('average ADE:', np.average(df['ADE']))
    print('average AIE:', np.average(df['AIE']))
    print('average time:', np.average(df['runtime']))

