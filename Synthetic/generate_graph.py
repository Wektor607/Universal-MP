import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from matplotlib.pyplot import cm
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class GraphGeneration():
    def __init__(self, m, n, emb_dim, graph_type, heterophily=False, homophily=False, feature_type='random'):
        self.m = m
        self.n = n
        self.emb_dim = emb_dim
        self.graph_type = graph_type
        self.heterophily = heterophily
        self.homophily = homophily
        self.feature_type = feature_type
    
    def plot_graph(self, G, pos=None, title="Graph", node_size=300, node_color='skyblue', with_labels=True):
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, edge_color='gray', font_size=10)
        plt.title(title)
        return plt

    def plot_color_graph(self, G, pos=None, title="Graph", node_size=300, with_labels=True):
        # Extract node labels
        node_labels = nx.get_node_attributes(G, 'label')
        
        # Determine unique labels and assign colors
        unique_labels = set(node_labels.values())
        print(node_labels)
        colors = ['red', 'black']  # Add more colors if you have more labels
        
        # Map node labels to colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        
        # Get node colors based on labels
        node_color = [color_map[node_labels[node]] for node in G.nodes]
        print(node_color)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, edge_color='gray', font_size=10)
        plt.title(title)
        return plt

    def plot_degree_histogram(self, G):
        """
        Plots the degree distribution histogram of a NetworkX graph G with improved visualization.

        Parameters:
        G (networkx.Graph): The input NetworkX graph.
        """
        # Calculate the degrees
        degrees = [val for (_, val) in G.degree()]
        
        # Get the unique degrees and their frequencies
        degree_counts = np.bincount(degrees)
        degrees_unique = np.arange(len(degree_counts))
        
        # Normalize the frequencies for colormap
        max_count = max(degree_counts)
        normalized_counts = np.array(degree_counts) / max_count if max_count > 0 else degree_counts
        colors = cm.Blues(normalized_counts)
        
        # Create the histogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(degrees_unique, degree_counts, color=colors, align='center')
        
        # Add labels and title
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution Histogram')
        ax.set_xlim(0, 10)
        
        # Show plot
        return fig

    def generate_node_features(self, G: nx.Graph):
        num_nodes = len(G.nodes)
        if self.feature_type == 'random':
            node_features = torch.randn(num_nodes, self.emb_dim)
        elif self.feature_type == 'one-hot':
            node_features = torch.eye(num_nodes)
        elif self.feature_type == 'degree':
            degree = [val for (_, val) in G.degree()]
            node_features = torch.tensor(degree, dtype=torch.float32).view(-1, 1)
            
        return node_features
    
    def rename_fields(self, data: Data):
        if self.heterophily or self.homophily:
            data.y = data.label.clone()
            del data.label
            
        data.edge_weight = data.weight.to(torch.float).clone()
        del data.weight
        
        return data
    
    def create_kagome_lattice(self):
        """ Create a Kagome lattice and return its NetworkX graph and positions. """
        G = nx.Graph()
        pos = {}
        
        def node_id(x, y, offset):
            return 2 * (x * self.n + y) + offset
        
        for x in range(self.m):
            for y in range(self.n):
                current_id0 = node_id(x, y, 0)
                current_id1 = node_id(x, y, 1)
                pos[current_id0] = (y, x)
                pos[current_id1] = (y + 0.5, x + 0.5)

                G.add_node(current_id0)
                G.add_node(current_id1)

                if y < self.n - 1:
                    right_id0 = node_id(x, y + 1, 0)
                    weight = random.uniform(0.1, 1.0)
                    G.add_edge(current_id0, right_id0, weight=weight)

                if x < self.m - 1:
                    down_id0 = node_id(x + 1, y, 0)
                    weight = random.uniform(0.1, 1.0)
                    G.add_edge(current_id0, down_id0, weight=weight)
        
        for node, position in pos.items():
            G.nodes[node]['pos'] = position
        
        return G, pos

    def create_square_grid(self):
        num_nodes = self.m * self.n
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        def node_id(x, y):
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

        # Generate positions for visualization
        pos = {(x * self.n + y): (y, x) for x in range(self.m) for y in range(self.n)}

        # Create NetworkX graph from adjacency matrix
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

    def create_triangle_grid(self):
        G = nx.triangular_lattice_graph(self.m, self.n)
        pos =  nx.get_node_attributes(G, 'pos')
        
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.1, 1.0)
        
        # Impossible
        # if self.heterophily:
        #     for node in G.nodes():
        #         x, y = node
        #         # Метка чередуется в зависимости от суммы координат
        #         G.nodes[node]['label'] = (x + y) % 2
        
        nodes = list(G.nodes())
        num_nodes = len(nodes)

        # Определяем количество узлов для каждой метки (половина узлов получает метку 0, половина — метку 1)
        half_nodes = num_nodes // 2

        # Назначаем метки узлам
        for i, node in enumerate(nodes):
            if i < half_nodes:
                G.nodes[node]['label'] = 0
            else:
                G.nodes[node]['label'] = 1
                
        return G, pos

    def create_hexagonal_grid(self):
        # Generate the hexagonal lattice graph
        G = nx.hexagonal_lattice_graph(self.m, self.n)
        pos = nx.get_node_attributes(G, 'pos')
        
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
                    node = (x, y)
                    if self.m < self.n:
                        G.nodes[node]['label'] = 0
                    else:
                        G.nodes[node]['label'] = 1
                    y += 1

        return G, pos

    def generate_graph(self):
        """
        Generates a PyG graph for nodes in a NetworkX graph based on their positions.

        Parameters:
        m (int): Number of rows in the grid graph.
        n (int): Number of columns in the grid graph.
        emb_dim (int): The dimension of the embeddings.

        Returns:
        Data: The generated graph with embeddings.
        """

        if self.graph_type == 'square_grid':
            G, pos = self.create_square_grid()
        elif self.graph_type == 'triangle':
            G, pos = self.create_triangle_grid()
        elif self.graph_type == 'hexagonal':
            G, pos = self.create_hexagonal_grid()
        elif self.graph_type == 'kagome':
            G, pos = self.create_kagome_lattice()
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type}")
        
        plt = self.plot_graph(G, pos, title=f"{self.graph_type} Graph")
        plt.savefig(f'/home/kit/aifb/cc7738/scratch/Universal-MP/Synthetic/pictures/{self.graph_type}_grid_graph.png')
        # plt.close()
        
        if self.homophily or self.heterophily:
            plt = self.plot_color_graph(G, pos, title=f"{self.graph_type} Graph")
            plt.savefig(f'/home/kit/aifb/cc7738/scratch/Universal-MP/Synthetic/pictures/{self.graph_type}_grid_graph_color.png')
        # plt.close()
        plt = self.plot_degree_histogram(G)
        plt.savefig(f'/home/kit/aifb/cc7738/scratch/Universal-MP/Synthetic/pictures/{self.graph_type}_grid_graph_degree_hist.png')
        # plt.close()
        
        # Converint G: networkx -> Data
        data = from_networkx(G)
        # print(data)
        data = self.rename_fields(data)
        weights = data.edge_weight.clone()
        
        # Creating node features
        data.x = self.generate_node_features(G)
        
        # Creating sparse adjacency matrix
        data = T.ToSparseTensor()(data)
        
        # Creating edge_index and return back edge_weight
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([col, row], dim=0)
        data.edge_weight = weights
        
        return data, data.adj_t, data.x, pos
