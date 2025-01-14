"""
Utils for generating random graph. Adopted from https://raw.githubusercontent.com/JiaruiFeng/KP-GNN/main/datasets/graph_generation.py
"""
import os
import sys
import math
import random
from enum import Enum

import networkx as nx
import os.path as osp
import numpy as np
import scipy.sparse as sp
from typing import *
import torch
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (to_undirected, 
                                coalesce, 
                                remove_self_loops,
                                from_networkx)
from data_utils import plot_graph
"""
    Generates random graphs of different types of a given size.
    Some of the graph are created using the NetworkX library, for more info see
    https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""


def erdos_renyi(N, degree, seed):
    """ Creates an Erdős-Rényi or binomial graph of size N with degree/N probability of edge creation """
    return nx.fast_gnp_random_graph(N, degree / N, seed, directed=False)


def barabasi_albert(N, degree, seed):
    """ Creates a random graph according to the Barabási–Albert preferential attachment model
        of size N and where nodes are atteched with degree edges """
    return nx.barabasi_albert_graph(N, degree, seed)



def caveman(N):
    """ Creates a caveman graph of m cliques of size k, with m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    G = nx.caveman_graph(m, N // m)
    return G


def tree(N, seed):
    """ Creates a tree of size N with a power law degree distribution """
    return nx.random_powerlaw_tree(N, seed=seed, tries=10000)


def ladder(N):
    """ Creates a ladder graph of N nodes: two rows of N/2 nodes, with each pair connected by a single edge.
        In case N is odd another node is attached to the first one. """
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def line(N):
    """ Creates a graph composed of N nodes in a line """
    return nx.path_graph(N)


def star(N):
    """ Creates a graph composed by one center node connected N-1 outer nodes """
    return nx.star_graph(N - 1)


def caterpillar(N, seed):
    """ Creates a random caterpillar graph with a backbone of size b (drawn from U[1, N)), and N − b
        pendent vertices uniformly connected to the backbone. """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, seed):
    """ Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
        from U[1, N − b ]) pendent vertices uniformly connected to the backbone, and additional
        N − b − p pendent vertices uniformly connected to the previous pendent vertices """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))
    return G


def random_grid_graph(N):
    """ Create a grid graph and return its NetworkX graph and positions. """

    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    G =  nx.grid_2d_graph(m, N // m)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    return G, pos


def randomize(A):
    """ Adds some randomness by toggling some edges without chancing the expected number of edges of the graph """
    BASE_P = 0.9

    # e is the number of edges, r the number of missing edges
    N = A.shape[0]
    e = np.sum(A) / 2
    r = N * (N - 1) / 2 - e

    # ep chance of an existing edge to remain, rp chance of another edge to appear
    if e <= r:
        ep = BASE_P
        rp = (1 - BASE_P) * e / r
    else:
        ep = BASE_P + (1 - BASE_P) * (e - r) / e
        rp = 1 - BASE_P

    array = np.random.uniform(size=(N, N), low=0.0, high=0.5)
    array = array + array.transpose()
    remaining = np.multiply(np.where(array < ep, 1, 0), A)
    appearing = np.multiply(np.multiply(np.where(array < rp, 1, 0), 1 - A), 1 - np.eye(N))
    ans = np.add(remaining, appearing)

    # assert (np.all(np.multiply(ans, np.eye(N)) == np.zeros((N, N))))
    # assert (np.all(ans >= 0))
    # assert (np.all(ans <= 1))
    # assert (np.all(ans == ans.transpose()))
    return ans


def init_nodefeats(G: nx.Graph,
                   feature_type: str,
                   emb_dim: int) -> torch.Tensor:
    """
    Input:
    ----------
    G : nx.Graph
        The NetworkX graph for which node features will be generated.

    Output:
    -------
    torch.Tensor
        A tensor containing the generated node features.

    Description:
    -----------
        This function generates node features for the graph based on the specified feature type.
        The features can be 'random', 'one-hot', based on the node 'degree'.
    """
    num_nodes: int = len(G.nodes)
    if feature_type == 'random':
        nodefeats: torch.Tensor = torch.randn(num_nodes, emb_dim)
    elif feature_type == 'one-hot':
        nodefeats: torch.Tensor = torch.eye(num_nodes)
    elif feature_type == 'degree':
        degree: List[int] = [val for (_, val) in G.degree()]
        nodefeats: torch.Tensor = torch.tensor(degree, dtype=torch.float32).view(-1, 1)
        
    return nodefeats
    
    
class RandomType(Enum):
    RANDOM = 0
    ERDOS_RENYI = 1
    BARABASI_ALBERT = 2
    GRID = 3
    CAVEMAN = 4
    TREE = 5
    LADDER = 6
    LINE = 7
    STAR = 8
    CATERPILLAR = 9
    LOBSTER = 10


# probabilities of each type in case of random type
MIXTURE = [(RandomType.ERDOS_RENYI, 0.2), (RandomType.BARABASI_ALBERT, 0.2), (RandomType.GRID, 0.05),
           (RandomType.CAVEMAN, 0.05), (RandomType.TREE, 0.15), (RandomType.LADDER, 0.05),
           (RandomType.LINE, 0.05), (RandomType.STAR, 0.05), (RandomType.CATERPILLAR, 0.1), (RandomType.LOBSTER, 0.1)]


def init_random_graph(N, type=RandomType.RANDOM, seed=None, degree=None):
    
    """
    Generates graphs of different types of a given size. Note:
     - graph are undirected and without weights on edges for random types
     - node values are sampled independently from U[0,1]
     - node features are initialized with the node degree, position or random values

    :param N:       number of nodes
    :param type:    type chosen between the categories specified in RandomType enum
    :param seed:    random seed
    :param degree:  average degree of a node, only used in some graph types
    :return:        adj_matrix: N*N numpy matrix
                    node_values: numpy array of size N
    """
    random.seed(seed)
    np.random.seed(seed)

    # sample which random type to use
    if type == RandomType.RANDOM:
        type = np.random.choice([t for (t, _) in MIXTURE], 1, p=[pr for (_, pr) in MIXTURE])[0]
        print(f"Randomly selected graph type: {type}")
        
    # generate the graph structure depending on the type
    if type == RandomType.ERDOS_RENYI:
        if degree == None: degree = random.random() * N
        G = erdos_renyi(N, degree, seed)
    elif type == RandomType.BARABASI_ALBERT:
        if degree == None: degree = int(random.random() * (N - 1)) + 1
        G = barabasi_albert(N, degree, seed)
    elif type == RandomType.GRID:
        G, pos = random_grid_graph(N)
    elif type == RandomType.CAVEMAN:
        G = caveman(N)
    elif type == RandomType.TREE:
        G = tree(N, seed)
    elif type == RandomType.LADDER:
        G = ladder(N)
    elif type == RandomType.LINE:
        G = line(N)
    elif type == RandomType.STAR:
        G = star(N)
    elif type == RandomType.CATERPILLAR:
        G = caterpillar(N, seed)
    elif type == RandomType.LOBSTER:
        G = lobster(N, seed)
    else:
        raise ValueError("Graph type not recognized")

    nodes = list(G)
    random.shuffle(nodes)
    adj_matrix = nx.to_numpy_array(G, nodes)
    adj_matrix = randomize(adj_matrix)
    node_values = np.random.uniform(low=0, high=1, size=N)
        

    plt.figure()
    try:
        plot = plot_graph(G, pos)
    except:
        plot = plot_graph(G)

    plot.savefig('draw.png')
        
    return G, adj_matrix, node_values, type


def random_edge_split(data: Data,
                undirected: bool,
                device: Union[str, int],
                val_pct: float,
                test_pct: float,
                split_labels: bool,
                include_negatives: bool = False) -> Dict[str, Data]:

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),

    ])
    train_data, val_data, test_data = transform(data)
    del train_data.neg_edge_label, train_data.neg_edge_label_index
    return {'train': train_data, 'valid': val_data, 'test': test_data}



def init_pyg_random(N: int, 
                    g_type: RandomType, 
                    seed: int, 
                    undirected = True, 
                    val_pct = 0.15,
                    test_pct = 0.05, 
                    split_labels = True, 
                    include_negatives = True) -> Data:
    
    G, _, _, _ = init_random_graph(N, g_type, seed=seed)

    data: Data = from_networkx(G)
    
    data.x = init_nodefeats(G, 'random', int(np.log(N)) + 16)

    data = T.ToSparseTensor()(data)
    
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
        undirected = True
    
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    
    if  undirected:
        data.edge_index = to_undirected(data.edge_index, data.edge_weight, reduce='add')[0]
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=float)

    data.adj_t = sp.csr_matrix((data.edge_weight.cpu(), (data.edge_index[0].cpu(), data.edge_index[1].cpu())), 
                shape=(data.num_nodes, data.num_nodes))

    splits = random_edge_split(data,
                undirected,
                'cpu',
                val_pct, # val_pct = 0.15
                test_pct, # test_pct =  0.5,
                split_labels, # split_labels = True,
                include_negatives)  # include_negatives = False

    # TODO save to .pt file
    return data, splits



if __name__ == '__main__':
    # TODO compare regular tilling with existiing graphs
    # params -> graph
    
    RANDOM = 0
    ERDOS_RENYI = 1
    BARABASI_ALBERT = 2
    GRID = 3
    SQUARE = 4
    CAVEMAN = 5
    TREE = 6
    LADDER = 7
    LINE = 8
    STAR = 9
    CATERPILLAR = 10
    LOBSTER = 11
    TRIANGULAR = 12
    HEXAGONAL = 13
    
    for i, g_type in enumerate([
                             RandomType.ERDOS_RENYI, 
                             RandomType.BARABASI_ALBERT, 
                             RandomType.GRID, 
                             RandomType.CAVEMAN, 
                             RandomType.TREE, 
                             RandomType.LADDER, 
                             RandomType.LINE, 
                             RandomType.STAR, 
                             RandomType.CATERPILLAR, 
                             RandomType.LOBSTER, 
                             ]):
        
        # G, adj_matrix, node_values, type = init_random_graph(40, g_type, seed=i)
        data, split = init_pyg_random(40, g_type, seed=i)
        
    # print(adj_matrix)
    print(data.x)
    print(data.edge_index)
    print(split.keys())
    print(split['train'])
    
    # graph -> split -> .pt 
    
    
    # graph statistic -> .csv, .tex
    
    
    