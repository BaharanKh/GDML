import os
import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
import torch


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


def cora_process(args):
    """
    Process the Cora dataset
    
    Parameters:
        args: Arguments containing processing options
        
    Returns:
        features: Processed feature matrix
        matrix: Adjacency matrix
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Cora')
    raw_data = pd.read_csv(os.path.join(data_dir, 'cora.content'), sep='\t', header=None)
    num = raw_data.shape[0]
    
    # Map from paper ID to index
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b, a)
    mapping = dict(c)

    features = raw_data.iloc[:, 1:-1]
    labels = pd.get_dummies(raw_data[1434])
    raw_data_cites = pd.read_csv(os.path.join(data_dir, 'cora.cites'), sep='\t', header=None)

    # Create adjacency matrix
    matrix = np.zeros((num, num))
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        x = mapping[i]
        y = mapping[j]
        matrix[x][y] = matrix[y][x] = 1

    arr = np.array(features)

    return arr, matrix


def pubmed_process(args):
    """
    Process the Pubmed dataset
    
    Parameters:
        args: Arguments containing processing options
        
    Returns:
        features: Processed feature matrix
        matrix: Adjacency matrix
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Pubmed')
    
    # Load preprocessed data from pickle files
    matrix = pkl.load(open(os.path.join(data_dir, 'matrix.pkl'), 'rb'))
    features = pkl.load(open(os.path.join(data_dir, 'features.pkl'), 'rb'))

    return features, matrix


def sbm_process(args):
    """
    Generate a synthetic network using stochastic block model
    
    Parameters:
        args: Arguments containing processing options
        
    Returns:
        matrix: Adjacency matrix
    """
    n_nodes = 3000
    n_comps = 200
    sizes = list(np.ones(n_comps, dtype=int) * (n_nodes // n_comps))
    
    # Create block probability matrix with higher probability within blocks
    probs = np.ones((n_comps, n_comps)) * 0.0001
    for i in range(n_comps):
        probs[i][i] = 0.75

    # Generate stochastic block model
    G = nx.stochastic_block_model(sizes, probs, seed=0)
    matrix = nx.adjacency_matrix(G).toarray()
    print('Number of edges in SBM graph: ', matrix.sum() // 2)
    
    return matrix


def find_focal_set(matrix):
    """
    Find a maximal independent set in the 2-hop graph (focal nodes)
    
    Parameters:
        matrix: Adjacency matrix
        
    Returns:
        maximal_set: Set of indices for focal nodes
    """
    maximal_set = set()
    idx = list(range(len(matrix)))
    remained_nodes = set(idx)
    
    while remained_nodes:
        if len(remained_nodes) % 10 == 0:
            print(len(remained_nodes))
            
        remove_nodes = set()
        
        node_idx = random.choice(list(remained_nodes))
        
        maximal_set.add(node_idx)
        remove_nodes.add(node_idx)
        
        # Get direct neighbors
        neighbors = list(np.where(matrix[node_idx] == 1)[0])
        remove_nodes.update(neighbors)
        
        # Get 2-hop neighbors
        for n in neighbors:
            n_neighbors = np.where(matrix[n] == 1)[0]
            remove_nodes.update(n_neighbors)
        
        # Remove processed nodes from remaining nodes
        remained_nodes = remained_nodes - remove_nodes
    
    return maximal_set


def find_train_split(splits, ind):
    """
    Find training indices from K-fold splits
    
    Parameters:
        splits: List of indices for each fold
        ind: Index of fold to exclude (test set)
        
    Returns:
        train: Combined indices for training
    """
    train = []
    for i, l in enumerate(splits):
        if i == ind:
            continue
        train += l
    return train