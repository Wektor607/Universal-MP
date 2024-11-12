import torch 
from torch_geometric.utils import from_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import torch.nn.functional as F


def plot_colored_graph(G, node_features, title="Graph with Colored Nodes", 
                       cmap="Blues", node_size=500, seed=42):

    # Set the layout for node positioning
    pos = nx.spring_layout(G, seed=seed)

    if isinstance(node_features, dict):
        node_colors = [node_features[node] for node in G.nodes]
    else:
        node_colors = node_features

    # Plot the graph
    plt.figure(figsize=(8, 6))

    nx.draw(G, pos, with_labels=True, 
            node_color=node_colors, cmap=cmap, 
            edge_color="black", node_size=node_size)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label="Node Feature")
    plt.title(title)
    plt.savefig('toy_mpnn.png')


def normalize_adj(edge_index, num_nodes=None, edge_weight=None, 
                  direction='sym', self_loops=True):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    if self_loops:
        fill_value = 1.
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    if direction == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif direction == 'row':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif direction == 'col':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = edge_weight * deg_inv[col]
    else:
        raise ValueError()

    return torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))


def init_node_feat(data, num_features=1):
    """
    Parameters:
    - data: PyTorch Geometric Data object
        The graph data to which the node features will be added.
    - num_features: int, optional
        Number of features per node (default is 3 for RGB-like features).
    """
    num_nodes = data.num_nodes
    half_nodes = num_nodes // 2  # Divide nodes into two halves

    # Initialize features: first half [0, 0, 0] (white), second half [1, 1, 1] (black)
    features = []
    for i in range(num_nodes):
        if i < half_nodes:
            features.append([0] * num_features)  # White
        else:
            features.append([1] * num_features)  # Black

    return torch.tensor(features, dtype=torch.float32)


# Example Usage
# Load the Karate Club graph and convert it to a PyTorch Geometric Data object
G = nx.karate_club_graph()
data = from_networkx(G)

# Initialize the node features
init_node_feat(data)

# Display the initialized node features
print("Initialized Node Features (data.x):\n", data.x)

# Example Usage
# Create a simple graph
G = nx.Graph()
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 8)]
G.add_edges_from(edges)

# Assign grayscale features to each node, 0 (white) to 1 (black)
node_features = {node: 0 if i < 4 else 1 for i, node in enumerate(G.nodes)}

# Plot the graph using the function
plot_colored_graph(G, node_features, 
                   title="Graph with Node Features as Colors", 
                   cmap="gray")


# Define node feature matrix H (8 nodes, 3 features each as an example)
H = np.array([
    [0, 0, 0],  
    [0, 0, 0],  
    [0, 0, 0], 
    [0, 0, 0],  
    [1, 1, 1],  
    [1, 1, 1],  
    [1, 1, 1], 
    [1, 1, 1],  
])


A = np.zeros((8, 8))
edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)]
for (i, j) in edges:
    A[i, j] = 1
    A[j, i] = 1  # Since the graph is undirected, make A symmetric

# Display H and A
print("Node Feature Matrix H:\n", H)
print("\nAdjacency Matrix A:\n", A)


data = from_networkx(G)

x = init_node_feat(data, num_features=3)
data.x = torch.tensor(x, dtype=torch.float32)
adj = normalize_adj(data.edge_index.long(), len(data.x), direction='sym', 
                    self_loops=True)
x = data.x.float()
for i in range(10):
    print(x)
    x = torch.spmm(adj, x.float())
    plot_colored_graph(G, x)
    
# TODO the color is not distinguishable 
# G = nx.karate_club_graph()
# data = from_networkx(G)

# x = init_node_feat(data, num_features=3)
# data.x = torch.tensor(x, dtype=torch.float32)
# adj = normalize_adj(data.edge_index.long(), len(data.x), direction='sym', 
#                     self_loops=True)
# x = data.x.float()
# for i in range(10):
#     print(x)
#     x = torch.spmm(adj, x.float())
#     if True:
#         out = F.normalize(x, p=2)
#     plot_colored_graph(G, x)