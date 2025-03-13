import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random
import sys
import os
import scipy.special as sp
import networkx as nx
import seaborn as sns
import time
from numpy.linalg import inv
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.loader import NeighborLoader

from data_generative_process import dgp
from data_preprocess import cora_process, pubmed_process, sbm_process, find_focal_set, find_train_split, CustomDataset
from models import accuracy, GIN




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
        data, df_true, X = dgp(matrix)

        # data.rename(columns={'Z': 'T'}, inplace=True)

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
    # df = pd.DataFrame(list(zip(ADEs, APEs, variances)), columns=['ADE', 'APE', 'VAR'])
    df = pd.DataFrame(list(zip(ADEs, APEs, variances, runtime)), columns=['ADE', 'APE', 'VAR', 'runtime'])
    df.to_csv(f'../results/{args.dataset}_{timestr}.csv', index=False)
    print(n_coverage_theta / trials, n_coverage_alpha / trials)

    print('average ADE:', np.average(df['ADE']))
    print('average APE:', np.average(df['APE']))
    print('average time:', np.average(df['runtime']))

