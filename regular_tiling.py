
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx



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


def create_grid_graph(m, n):
    """ Create a grid graph and return its NetworkX graph and positions. """
    G = nx.grid_2d_graph(m, n)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    return G, pos


def create_square_grid(m, n):
    num_nodes = m * n
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    def node_id(x, y):
        return x * n + y
    
    for x in range(m):
        for y in range(n):
            current_id = node_id(x, y)
            
            # Right neighbor
            if y < n - 1:
                right_id = node_id(x, y + 1)
                adj_matrix[current_id, right_id] = 1
                adj_matrix[right_id, current_id] = 1
                
            # Down neighbor
            if x < m - 1:
                down_id = node_id(x + 1, y)
                adj_matrix[current_id, down_id] = 1
                adj_matrix[down_id, current_id] = 1

    # Generate positions for visualization
    pos = {(x * n + y): (y, x) for x in range(m) for y in range(n)}

    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    return G, pos


def create_triangle_grid(m, n):
    G = nx.triangular_lattice_graph(m, n)
    pos =  nx.get_node_attributes(G, 'pos')
    return G, pos


def create_hexagonal_grid(m, n):
    # Generate the hexagonal lattice graph
    G = nx.hexagonal_lattice_graph(m, n)
    pos = nx.get_node_attributes(G, 'pos')
    return G, pos


def plot_degree_histogram(G):
    """
    Plots the degree distribution histogram of a NetworkX graph G with improved visualization.

    Parameters:
    G (networkx.Graph): The input NetworkX graph.
    """
    # Calculate the degrees
    degrees = [val for (node, val) in G.degree()]
    
    # Get the unique degrees and their frequencies
    degree_counts = np.bincount(degrees)
    degrees_unique = np.arange(len(degree_counts))
    
    # Normalize the frequencies for colormap
    max_count = max(degree_counts)
    normalized_counts = np.array(degree_counts) / max_count if max_count > 0 else degree_counts
    colors = cm.Blues(normalized_counts)
    
    # Create the histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(degrees_unique, degree_counts, color=colors, align='center')
    
    # Add labels and title
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution Histogram')
    ax.set_xlim(0, 10)
    # Show plot
    plt.show()
    
    # Show plot
    return plt


def plot_graph(G, pos=None, title="Graph", node_size=300, node_color='skyblue', with_labels=True):
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, edge_color='gray', font_size=10)
    plt.title(title)
    return plt


def gen_pyg_from_nx(m, n, emb_dim=32, graph_type='grid'):
    """
    Generates a PyG graph for nodes in a NetworkX graph based on their positions.

    Parameters:
    m (int): Number of rows in the grid graph.
    n (int): Number of columns in the grid graph.
    emb_dim (int): The dimension of the embeddings.

    Returns:
    Data: The generated graph with embeddings.
    """
   
    if graph_type == 'grid':
        G, pos = create_grid_graph(m, n)
    elif graph_type == 'triangle':
        G, pos = create_triangle_grid(m, n)
    elif graph_type == 'hexagonal':
        G, pos = create_hexagonal_grid(m, n)
    elif graph_type == 'kagome':
        G, pos = create_kagome_lattice(m, n)
    else:
        raise ValueError(f"Invalid graph type: {graph_type}")
    
    data = from_networkx(G)
    
    emb_layer = torch.nn.Embedding(data.num_nodes, emb_dim)
    
    pos_list = []
    for i, x in pos.items():
        pos_list.append(x[0] * m + x[1])
    
    pos_array = np.asarray(pos_list)
    pos_array = np.clip(pos_array, 0, data.num_nodes - 1)
    with torch.no_grad():
        pos_tensor = torch.tensor(pos_array, dtype=torch.int64)
        vectors = emb_layer(pos_tensor)
    data.x = vectors

    return data, G, pos


# main function
if __name__ == "__main__":
    # Create a grid graph
    m, n = 5, 5
    dim_feat = 32
    graph_type = 'kagome'
    for graph_type in ['grid', 'triangle', 'hexagonal', 'kagome']:
        data, G, pos = gen_pyg_from_nx(m, n, dim_feat, graph_type)
        
        # plt = plot_graph(G, pos, title="Grid Graph")
        # plt.savefig('grid_graph.png')
        # plt.close()
        
        # plt = plot_degree_histogram(G)
        # plt.savefig('grid_graph_degree_hist.png')
        # plt.close()
        
        print(data.x)
        print(data.edge_index)


        