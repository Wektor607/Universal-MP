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
from typing import *
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from matplotlib.figure import Figure
from torch_geometric.data import Data, Dataset, InMemoryDataset
from graph_generation import generate_graph, GraphType


from torch_geometric.utils import coalesce, to_undirected, from_networkx

"""
    Generates random graphs of different types of a given size.
    Some of the graph are created using the NetworkX library, for more info see
    https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        N: int=10000,
    ):
        self.dataset_name = name
        N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{N}.pt'

    def process(self):
        graph_type_str = f"GraphType.{self.dataset_name}"
        nx_data = generate_graph(N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])


class GraphType(Enum):
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



# probabilities of each type in case of random type
MIXTURE = [(GraphType.ERDOS_RENYI, 0.2), (GraphType.BARABASI_ALBERT, 0.2), (GraphType.GRID, 0.05),
           (GraphType.CAVEMAN, 0.05), (GraphType.TREE, 0.15), (GraphType.LADDER, 0.05),
           (GraphType.LINE, 0.05), (GraphType.STAR, 0.05), (GraphType.CATERPILLAR, 0.1), (GraphType.LOBSTER, 0.1)]


def erdos_renyi(N, degree, seed):
    """ Creates an Erdős-Rényi or binomial graph of size N with degree/N probability of edge creation """
    return nx.fast_gnp_random_graph(N, degree / N, seed, directed=False)


def barabasi_albert(N, degree, seed):
    """ Creates a random graph according to the Barabási–Albert preferential attachment model
        of size N and where nodes are atteched with degree edges """
    return nx.barabasi_albert_graph(N, degree, seed)


def grid(N):
    """ Creates a m x k 2d grid graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.grid_2d_graph(m, N // m)


def triangular(N):
    """ Creates a m x k 2d grid triangular graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.triangular_lattice_graph(m, N // m)


def hexagonal(N):
    """ Creates a m x k 2d grid hexagonal graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.hexagonal_lattice_graph(m, N // m)


def caveman(N):
    """ Creates a caveman graph of m cliques of size k, with m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.caveman_graph(m, N // m)


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


def square_grid(M, N, seed) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Output:
    -------
    Tuple[nx.Graph, Dict[int, Tuple[float, float]]]
        A tuple containing the square grid graph and the node positions.

    Description:
    -----------
        This function generates a square grid graph with m rows and n columns.
        It assigns 'random edge weights' from Uniform distribution and optional node labels based on heterophily or homophily settings.

    """
    #TODO to be classified within this graph class
    num_nodes: int = M * N
    adj_matrix: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=int)
    
    def node_id(x: int, y: int) -> int:
        return x * N + y
    
    for x in range(M):
        for y in range(N):
            current_id = node_id(x, y)
            
            # Right neighbor
            if y < N - 1:
                right_id = node_id(x, y + 1)
                adj_matrix[current_id, right_id] = 1
                adj_matrix[right_id, current_id] = 1
                
            # Down neighbor
            if x < M - 1:
                down_id = node_id(x + 1, y)
                adj_matrix[current_id, down_id] = 1
                adj_matrix[down_id, current_id] = 1

    pos: Dict[int, Tuple[float, float]] = {(x * N + y): (y, x) for x in range(M) for y in range(N)}

    G = nx.from_numpy_array(adj_matrix)
    
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 1.0)
    
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
    
    return G, pos


def create_kagome_lattice(m, n):
    """ Create a Kagome lattice and return its NetworkX graph and positions. """
    G = nx.Graph()
    pos = {}
    
    def node_id(x, y, offset):
        return 2 * (x * n + y) + offset
    
    for x in range(m):
        for y in range(n):
            # Two nodes per cell (offset 0 and 1)
            current_id0 = node_id(x, y, 0)
            current_id1 = node_id(x, y, 1)
            pos[current_id0] = (y, x)
            pos[current_id1] = (y + 0.5, x + 0.5)
            
            # Add nodes
            G.add_node(current_id0)
            G.add_node(current_id1)
            
            # Right and down connections
            if y < n - 1:
                right_id0 = node_id(x, y + 1, 0)
                right_id1 = node_id(x, y + 1, 1)
                G.add_edge(current_id0, right_id0)
                G.add_edge(right_id1, right_id0)
                G.add_edge(right_id0, current_id0)
                G.add_edge(right_id0, right_id1)
                
            if x < m - 1:
                down_id0 = node_id(x + 1, y, 0)
                down_id1 = node_id(x + 1, y, 1)
                G.add_edge(current_id0, down_id0)
                G.add_edge(current_id1, down_id1)
                G.add_edge(down_id0, current_id0)
                G.add_edge(down_id1, current_id1)
            
            # Diagonal connections
            if x < m - 1 and y < n - 1:
                diag_id0 = node_id(x + 1, y + 1, 0)
                diag_id1 = node_id(x + 1, y + 1, 1)
                G.add_edge(current_id1, diag_id0)
                G.add_edge(diag_id0, current_id1)
                G.add_edge(current_id1, diag_id1)
                G.add_edge(diag_id1, current_id1)
    
    return G, pos



def init_nodefeats(self, G: nx.Graph) -> torch.Tensor:
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
    if self.feature_type == 'random':
        nodefeats: torch.Tensor = torch.randn(num_nodes, self.emb_dim)
    elif self.feature_type == 'one-hot':
        nodefeats: torch.Tensor = torch.eye(num_nodes)
    elif self.feature_type == 'degree':
        degree: List[int] = [val for (_, val) in G.degree()]
        nodefeats: torch.Tensor = torch.tensor(degree, dtype=torch.float32).view(-1, 1)
        
    return nodefeats


def generate_graph(N, type=GraphType.RANDOM, seed=None, degree=None):
    """
    Generates random graphs of different types of a given size. Note:
     - graph are undirected and without weights on edges
     - node values are sampled independently from U[0,1]

    :param N:       number of nodes
    :param type:    type chosen between the categories specified in GraphType enum
    :param seed:    random seed
    :param degree:  average degree of a node, only used in some graph types
    :return:        adj_matrix: N*N numpy matrix
                    node_values: numpy array of size N
    """
    random.seed(seed)
    np.random.seed(seed)

    # sample which random type to use
    if type == GraphType.RANDOM:
        type = np.random.choice([t for (t, _) in MIXTURE], 1, p=[pr for (_, pr) in MIXTURE])[0]

    # generate the graph structure depending on the type
    if type == GraphType.ERDOS_RENYI:
        if degree == None: degree = random.random() * N
        G = erdos_renyi(N, degree, seed)
    elif type == GraphType.BARABASI_ALBERT:
        if degree == None: degree = int(random.random() * (N - 1)) + 1
        G = barabasi_albert(N, degree, seed)
    elif type == GraphType.GRID:
        G = grid(N)
    elif type == GraphType.TRIANGULAR:
        G = triangular(N)
    elif type == GraphType.HEXAGONAL:
        G = hexagonal(N)
    elif type == GraphType.CAVEMAN:
        G = caveman(N)
    elif type == GraphType.TREE:
        G = tree(N, seed)
    elif type == GraphType.LADDER:
        G = ladder(N)
    elif type == GraphType.LINE:
        G = line(N)
    elif type == GraphType.STAR:
        G = star(N)
    elif type == GraphType.CATERPILLAR:
        G = caterpillar(N, seed)
    elif type == GraphType.LOBSTER:
        G = lobster(N, seed)
    elif type == GraphType.SQUARE:
        G = square_grid(N, seed)
    else:
        raise ValueError("Graph type not recognized")

    # generate adjacency matrix and nodes values
    if type == GraphType.RANDOM:
        nodes = list(G)
        random.shuffle(nodes)
        adj_matrix = nx.to_numpy_array(G, nodes)
        node_values = np.random.uniform(low=0, high=1, size=N)
        adj_matrix = randomize(adj_matrix)
    else:
        node_values = init_nodefeats(G)
        
    # draw the graph created
    # import pdb; pdb.set_trace()
    # nx.draw(G, pos=nx.spring_layout(G))
    # plt.draw()

    return adj_matrix, node_values, type


def calc_cn(self, ei: torch.Tensor, adj: 'scipy.sparse.csr_matrix', bs: int = 10000) -> torch.FloatTensor:
    """
    Computes the number of common neighbors for node pairs in `ei` using adjacency matrix `adj`.

    Input:
    - ei (torch.Tensor): Shape (2, E), where E is the number of edges.
    - adj ('scipy.sparse.csr_matrix'): Graph adjacency matrix in CSR format.
    - bs (int): Batch size for processing node pairs. Default is 10000.

    Output:
    - torch.FloatTensor: Tensor containing the number of common neighbors for each pair of nodes in `ei`.
    """
    ll = torch.utils.data.DataLoader(range(ei.size(1)), bs)
    sc = []

    for idx in ll:
        s, d = ei[0, idx].to('cpu'), ei[1, idx].to('cpu')
        cur_sc = np.array(adj[s].multiply(adj[d]).sum(axis=1)).flatten()
        sc.append(cur_sc)

    sc = np.concatenate(sc, axis=0)
    return torch.FloatTensor(sc)


def plot_cn_dist(self, tr_cn: torch.FloatTensor, te_cn: torch.FloatTensor) -> None:
    """
    Generates a density plot of the number of common neighbors for training and testing datasets.

    Input:
    - tr_cn (torch.FloatTensor): Tensor with common neighbors for the training dataset.
    - te_cn (torch.FloatTensor): Tensor with common neighbors for the testing dataset.

    Output:
    - None: Saves the plot as a PNG file.

    Description:
    This function uses Kernel Density Estimation (KDE) via Seaborn to visualize the distribution of common neighbors
    for both training and testing datasets.
    """
    # Convert tensors to NumPy arrays
    tr_cn = tr_cn.to('cpu').numpy()
    te_cn = te_cn.to('cpu').numpy()

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(tr_cn, label='Train', color='red', linewidth=2)
    sns.kdeplot(te_cn, label='Test', color='blue', linewidth=2)
    plt.xlabel('Common Neighbors')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Common Neighbors Distribution')
    plt.savefig(f'{FILE_PATH}/cn_dist.png')


def plot_color_graph(self, G: nx.Graph, pos: Optional[Dict[int, Tuple[float, float]]] = None, 
                        title: str = "Graph", node_size: int = 300, with_labels: bool = True) -> Figure:
    """
    Input:
    ----------
    G : nx.Graph
        The NetworkX graph to be plotted.
    pos : Optional[Dict[int, Tuple[float, float]]], default=None
        A dictionary of node positions, where keys are node indices and values are (x, y) coordinates.
    title : str, default="Graph"
        The title of the graph plot.
    node_size : int, default=300
        The size of the nodes in the plot.
    with_labels : bool, default=True
        Whether or not to display labels on the nodes.

    Output:
    -------
    Figure
        The matplotlib figure object representing the final graph plot.

    Description:
    -----------
        This function plots a colored graph where each node's color represents its label.
    """
    node_labels: Dict[int, Any] = nx.get_node_attributes(G, 'label')
    
    unique_labels: Set[Any] = set(node_labels.values())
    
    colors: List[str] = ['red', 'black']  # Add more colors if you have more labels
    
    color_map: Dict[Any, str] = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    node_color: List[str] = [color_map[node_labels[node]] for node in G.nodes]
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, 
            edge_color='gray', font_size=10)
    plt.title(title)
    return plt.gcf()


def plot_degree_histogram(self, G: nx.Graph) -> Figure:
    """
    Input:
    ----------
    G : nx.Graph
        The NetworkX graph whose degree distribution will be plotted.

    Output:
    -------
    Figure
        The matplotlib figure object representing the histogram plot.

    Description:
    -----------
        This function plots a histogram of the node degree frequency distribution for a graph.

    """
    
    degrees: List[int] = [val for (_, val) in G.degree()]
    
    # Get the unique degrees and their frequencies
    degree_counts: np.ndarray = np.bincount(degrees)
    degrees_unique: np.ndarray = np.arange(len(degree_counts))
    
    # Normalize the frequencies for colormap
    max_count: int = max(degree_counts)
    normalized_counts: np.ndarray = np.array(degree_counts) / max_count if max_count > 0 else degree_counts
    colors = cm.Blues(normalized_counts)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(degrees_unique, degree_counts, color=colors, align='center')
    
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution Histogram')
    ax.set_xlim(0, 10)
    
    return fig



if __name__ == '__main__':
    # TODO compare regular tilling with existiing graphs
    # params -> graph
    for i in range(100):
        adj_matrix, node_values, type = generate_graph(10, GraphType.RANDOM, seed=i)
    print(adj_matrix)

    pos = None
    plt = plot_color_graph(G, pos, title="Grid Graph")
    plt.savefig('grid_graph.png')
    plt.close()
    
    # graph -> split -> .pt 
    
    
    # graph statistic -> .csv, .tex
    
    
    