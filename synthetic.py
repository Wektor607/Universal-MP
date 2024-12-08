from typing import *
import os
import random
import networkx as nx
import argparse
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import torch
from matplotlib.pyplot import cm
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx, to_undirected


class SyntheticGraphGeneration:
    SQUARE = 0
    TRIANGLE = 1
    HEXAGONAL = 2
    FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(FILE_PATH)
    
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


    def create_grid_graph(m, n):
        """ Create a grid graph and return its NetworkX graph and positions. """
        G = nx.grid_2d_graph(m, n)
        pos = {(x, y): (x, y) for x, y in G.nodes()}
        return G, pos

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
        
        # TODO - add node feature initialization method 
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
        data.x = self.generate_nodefeats(G)
        
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


class arg_params:
    def __init__(self):
        self.m = 4  # Number of rows in the grid
        self.n = 4  # Number of columns in the grid
        self.emb_dim = 32  # Embedding dimension
        self.graph_type = 2  # Type of graph structure: 0 = Square, 1 = Triangle, 2 = Hexagonal
        self.heterophily = False  # Enable heterophily for the graph
        self.homophily = True  # Enable homophily for the graph
        self.feature_type = 'random'  # Embedding generation type: 'random', 'one-hot', or 'degree'
        self.undirected = True  # Specify if the graph is undirected
        self.device = 'cpu'  # Device to run computations (e.g., 'cpu' or 'cuda:0')
        self.val_pct = 0.2  # Percentage of edges for validation
        self.test_pct = 0.1  # Percentage of edges for testing
        self.split_labels = True  # Include labels for edge splits
        self.include_negatives = True  # Include negative edge samples


# main function
if __name__ == "__main__":
    # Create a grid graph
    args = arg_params()
    
    gen_graph = SyntheticGraphGeneration(
        m=args.m,
        n=args.n,
        emb_dim=args.emb_dim,
        graph_type=args.graph_type,
        heterophily=args.heterophily,
        homophily=args.homophily,
        feature_type=args.feature_type,
        cfg=args
    )
    
    data, adj_matrix, nodefeats, splits = gen_graph.generate_graph()
    print(splits)
    print(data, '\n')
    print(data.pos, '\n')
    print(adj_matrix, '\n')