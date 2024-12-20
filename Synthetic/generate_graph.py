import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from typing import *
import random
import numpy as np
import networkx as nx
import argparse
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.figure import Figure
import scipy.sparse as sp

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx, to_undirected

class SyntheticGraphGeneration:
    SQUARE = 0
    TRIANGLE = 1
    HEXAGONAL = 2
    FILE_PATH = '/home/kit/aifb/cc7738/scratch/Universal-MP/Synthetic/pictures'
    
    def __init__(self, m: int, n: int, emb_dim: int, graph_type: int, heterophily: bool=False, 
                 homophily: bool=False, feature_type: str='random', cfg: argparse.Namespace=None):
        self.m            = m
        self.n            = n
        self.emb_dim      = emb_dim
        self.graph_type   = graph_type
        self.heterophily  = heterophily
        self.homophily    = homophily
        self.feature_type = feature_type
        self.cfg          = cfg

    def plot_graph(self, G: nx.Graph, pos: Optional[Dict[int, Tuple[float, float]]] = None,
                   title: str = "Graph", node_size: int = 300, node_color: str = 'skyblue', 
                   with_labels: bool = True) -> Figure:
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
        node_color : str, default='skyblue'
            The color of the nodes in the plot.
        with_labels : bool, default=True
            Whether or not to display labels on the nodes.

        Output:
        -------
        Figure
            The matplotlib figure object representing the final graph plot.

        Description:
        -----------
            This function plots the structure of the given graph using NetworkX and matplotlib.
            It allows for customization of the title, node size, node color, and the option to include labels.
        """
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, 
                edge_color='gray', font_size=10)
        plt.title(title)
        return plt.gcf()

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

    def calculate_common_neighbors(self, edge_index: torch.Tensor, A: 'scipy.sparse.csr_matrix', batch_size: int = 10000) -> torch.FloatTensor:
        """
        Input:
        - edge_index (torch.Tensor): A tensor of shape (2, E), where E is the number of edges. 
        - A ('scipy.sparse.csr_matrix'): The adjacency matrix of the graph in CSR (Compressed Sparse Row) format.
        - batch_size (int): The batch size for processing pairs of nodes. Default is 10000.

        Output:
        - torch.FloatTensor: A tensor containing the number of common neighbors for each pair of nodes in `edge_index`.

        Description:
            This function computes the number of common neighbors for each pair of nodes in `edge_index` using the adjacency matrix `A`.
        """
        link_loader = torch.utils.data.DataLoader(range(edge_index.size(1)), batch_size)
        scores = []

        for ind in link_loader:
            src, dst = edge_index[0, ind].to('cpu'), edge_index[1, ind].to('cpu')
            cur_scores = np.array(A[src].multiply(A[dst]).sum(axis=1)).flatten()
            scores.append(cur_scores)

        scores = np.concatenate(scores, axis=0)
        return torch.FloatTensor(scores)

    def plot_common_neighbors_distribution(self, train_cn: torch.FloatTensor, test_cn: torch.FloatTensor) -> None:
        """
        Input:
        - train_cn (torch.FloatTensor): A tensor containing the number of common neighbors for the training dataset.
        - test_cn (torch.FloatTensor): A tensor containing the number of common neighbors for the testing dataset.

        Output:
        - None: The function saves the plot of the distribution as a PNG file.

        Description:
        This function generates a density plot of the number of common neighbors for both the training and testing datasets
        using Kernel Density Estimation (KDE) with the Seaborn library. It visualizes the distribution and saves the plot
        to a specified file path.
        """

        # Convert tensors to NumPy arrays for plotting
        train_cn = train_cn.to('cpu').numpy()
        test_cn = test_cn.to('cpu').numpy()
        print(train_cn)
        plt.figure(figsize=(10, 6))
        sns.kdeplot(train_cn, label='Training', color='red', linewidth=2)
        sns.kdeplot(test_cn, label='Testing', color='blue', linewidth=2)
        plt.xlabel('Number of Common Neighbors')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Distribution of Common Neighbors')
        plt.savefig(f'{SyntheticGraphGeneration.FILE_PATH}/common_neighbours_distribution.png')

    def generate_node_features(self, G: nx.Graph) -> torch.Tensor:
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
            node_features: torch.Tensor = torch.randn(num_nodes, self.emb_dim)
        elif self.feature_type == 'one-hot':
            node_features: torch.Tensor = torch.eye(num_nodes)
        elif self.feature_type == 'degree':
            degree: List[int] = [val for (_, val) in G.degree()]
            node_features: torch.Tensor = torch.tensor(degree, dtype=torch.float32).view(-1, 1)
            
        return node_features

    def rename_fields(self, data: Data) -> Data:
        """
        Input:
        ----------
        data : Data
            The PyTorch Geometric Data object representing the graph.

        Output:
        -------
        Data
            The modified Data object with renamed fields.

        Description:
        -----------
            This function renames certain fields in the Data object for consistency.
        """
        
        if self.heterophily or self.homophily:
            data.y = data.label.clone()
            del data.label
            
        data.edge_weight = data.weight.to(torch.float).clone()
        del data.weight
        
        return data

    def create_square_grid(self) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
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
        num_nodes: int = self.m * self.n
        adj_matrix: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=int)
        
        def node_id(x: int, y: int) -> int:
            return x * self.n + y
        
        for x in range(self.m):
            for y in range(self.n):
                current_id = node_id(x, y)
                
                # Right neighbor
                if y < self.n - 1:
                    right_id = node_id(x, y + 1)
                    adj_matrix[current_id, right_id] = 1
                    adj_matrix[right_id, current_id] = 1
                    
                # Down neighbor
                if x < self.m - 1:
                    down_id = node_id(x + 1, y)
                    adj_matrix[current_id, down_id] = 1
                    adj_matrix[down_id, current_id] = 1

        pos: Dict[int, Tuple[float, float]] = {(x * self.n + y): (y, x) for x in range(self.m) for y in range(self.n)}

        G = nx.from_numpy_array(adj_matrix)
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.1, 1.0)
        
        for node, position in pos.items():
            G.nodes[node]['pos'] = position
        
        # Assign labels in a checkerboard pattern
        if self.heterophily:
            for x in range(self.m):
                for y in range(self.n):
                    current_id = node_id(x, y)
                    G.nodes[current_id]['label'] = (x + y) % 2
        
        if self.homophily:
            for x in range(self.m):
                for y in range(self.n):
                    current_id = node_id(x, y)
                    if current_id >= len(G.nodes) / 2:
                        G.nodes[current_id]['label'] = 1
                    else:
                        G.nodes[current_id]['label'] = 0
        return G, pos

    def create_triangle_grid(self) -> Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[float, float]]]:
        """
        Output:
        -------
        Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[float, float]]]
            A tuple containing the triangular grid graph and the node positions.

        Description:
        -----------
            This function creates a triangular grid graph using NetworkX.
            It assigns 'random edge weights' from Uniform distribution and optional node labels based on homophily settings.
            **Note:** Heterophily is not applicable for this type of grid; only homophily labeling is supported.

        """
        if self.heterophily:
            raise ValueError("Heterophily is not supported for the 'triangle' graph type.")
    
        G = nx.triangular_lattice_graph(self.m, self.n)
        pos: Dict[Tuple[int, int], Tuple[float, float]] = nx.get_node_attributes(G, 'pos')
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.1, 1.0)
        
        nodes: List[Tuple[int, int]] = list(G.nodes())
        num_nodes: int = len(nodes)

        if self.homophily:
            half_nodes: int = num_nodes // 2

            # Assign labels to nodes
            for i, node in enumerate(nodes):
                if i < half_nodes:
                    G.nodes[node]['label'] = 0
                else:
                    G.nodes[node]['label'] = 1
                
        return G, pos

    def create_hexagonal_grid(self) -> Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[float, float]]]:
        """
        Output:
        -------
        Tuple[nx.Graph, Dict[Tuple[int, int], Tuple[float, float]]]
            A tuple containing the hexagonal grid graph and the node positions.

        Description:
        -----------
            This function generates a hexagonal grid graph using NetworkX.
            It assigns 'random edge weights' from Uniform distribution and optional node labels based on heterophily or homophily settings.

        """
        G = nx.hexagonal_lattice_graph(self.m, self.n)
        pos: Dict[Tuple[int, int], Tuple[float, float]] = nx.get_node_attributes(G, 'pos')
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.1, 1.0)

        if self.heterophily:
            for node in G.nodes:
                id1, id2 = node
                G.nodes[node]['label'] = (id1 + id2) % 2
        
        if self.homophily:
            for node in G.nodes:
                x, _ = node
                if x >= (max(self.m, self.n) / 2):
                    G.nodes[node]['label'] = 1
                else:
                    G.nodes[node]['label'] = 0
            
            # Solves the problem with painting in the center of the network
            if self.m % 2 == 0 or self.n % 2 == 0:
                x = max(self.m, self.n) // 2
                y = 0
                while y < ((self.m * 2 + 1) / 2):
                    node: Tuple[int, int] = (x, y)
                    if self.m < self.n:
                        G.nodes[node]['label'] = 0
                    else:
                        G.nodes[node]['label'] = 1
                    y += 1

        return G, pos

    def get_edge_split(self, data: Data,
                   undirected: bool,
                   device: Union[str, int],
                   val_pct: float,
                   test_pct: float,
                   split_labels: bool,
                   include_negatives: bool = False) -> Dict[str, Data]:
        """
        Input:
        -----------
        data : Data
            The input graph data in the PyTorch Geometric format.
        undirected : bool
            Specifies whether the graph is undirected. If True, edges will be treated as bidirectional.
        device : Union[str, int]
            The device to which the data will be moved (e.g., 'cpu', 'cuda', or a specific GPU ID).
        val_pct : float
            The percentage of edges to be used for validation.
        test_pct : float
            The percentage of edges to be used for testing.
        split_labels : bool
            If True, includes the labels for split edges (positive and negative) in the returned datasets.
        include_negatives : bool, default=False
            If True, includes negative samples (non-existent edges) in the training data for better evaluation of link prediction.

        Returns:
        --------
        Dict[str, Data]
            A dictionary containing the split data:
            - 'train': The training dataset.
            - 'valid': The validation dataset.
            - 'test': The test dataset.

        Description:
        ------------
            This function splits the dataset edges into training, validation, and test sets.
        """
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            RandomLinkSplit(is_undirected=undirected,
                            num_val=val_pct,
                            num_test=test_pct,
                            add_negative_train_samples=include_negatives,
                            split_labels=split_labels),

        ])
        # del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
        train_data, val_data, test_data = transform(data)
        # Delete negative samples from train dataset
        del train_data.neg_edge_label, train_data.neg_edge_label_index
        return {'train': train_data, 'valid': val_data, 'test': test_data}

    def generate_graph(self) -> Tuple[Data, torch.Tensor, torch.Tensor, Dict[int, Tuple[float, float]]]:
        """
        Output:
        -------
        Tuple[Data, torch.Tensor, torch.Tensor, Dict[int, Tuple[float, float]]]
            A tuple containing the generated PyG graph, adjacency matrix, node features, and node positions.

        Description:
        -----------
            This function generates a PyTorch Geometric graph based on the specified graph type.
            It creates the graph, visualizes it, and saves the plots. It also creates node features and sparse adjacency matrices.

        """
        if self.graph_type == SyntheticGraphGeneration.SQUARE:
            G, pos = self.create_square_grid()
        elif self.graph_type == SyntheticGraphGeneration.TRIANGLE:
            G, pos = self.create_triangle_grid()
        elif self.graph_type == SyntheticGraphGeneration.HEXAGONAL:
            G, pos = self.create_hexagonal_grid()
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}")
        
        plt = self.plot_graph(G, pos, title=f"{self.graph_type} Graph")
        plt.savefig(f'{SyntheticGraphGeneration.FILE_PATH}/{self.graph_type}_grid_graph.png')
        
        if self.homophily or self.heterophily:
            plt = self.plot_color_graph(G, pos, title=f"{self.graph_type} Graph")
            plt.savefig(f'{SyntheticGraphGeneration.FILE_PATH}/{self.graph_type}_grid_graph_color.png')
        
        plt = self.plot_degree_histogram(G)
        plt.savefig(f'{SyntheticGraphGeneration.FILE_PATH}/{self.graph_type}_grid_graph_degree_hist.png')
        
        # Convert G: networkx -> Data
        data: Data = from_networkx(G)
        data.weight = torch.ones(data.edge_index.size(1), dtype=float)
        data = self.rename_fields(data)
        weights: torch.Tensor = data.edge_weight.clone()
        
        # Creating node features
        data.x = self.generate_node_features(G)
        
        # Creating sparse adjacency matrix
        data = T.ToSparseTensor()(data)
        
        # Creating edge_index and return back edge_weight
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([col, row], dim=0)
        
        if self.cfg.undirected:
            edge_index = to_undirected(data.edge_index, weights, reduce='add')
            data.edge_index = edge_index[0]
            data.edge_weight = weights #edge_index[1]
            data.adj_t = sp.csr_matrix((data.edge_weight.cpu(), (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
                            shape=(data.num_nodes, data.num_nodes))
            # data.adj_t = SparseTensor(row=data.edge_index[0], 
            #                           col=data.edge_index[1], 
            #                           value=data.edge_weight.to(torch.float32))
        else:
            data.adj_t = sp.csr_matrix((weights.cpu(), (data.edge_index[0].cpu(), data.edge_index[1].cpu())), 
                     shape=(data.num_nodes, data.num_nodes))
            data.edge_weight = weights
        
        # Splitting edges
        splits = self.get_edge_split(data,
                   self.cfg.undirected,
                   self.cfg.device,
                   self.cfg.val_pct,
                   self.cfg.test_pct,
                   self.cfg.split_labels,
                   self.cfg.include_negatives)
        
        train_edge_index = splits['train']['pos_edge_label_index']
        test_edge_index = splits['test']['pos_edge_label_index']

        # Calculate common neighbors
        train_cn = self.calculate_common_neighbors(train_edge_index, data.adj_t)
        test_cn = self.calculate_common_neighbors(test_edge_index, data.adj_t)

        # Plot common neighbors distribution
        self.plot_common_neighbors_distribution(train_cn, test_cn)
        print(data.num_nodes)
        return data, data.adj_t, data.x, splits
