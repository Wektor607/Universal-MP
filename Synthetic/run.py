import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join( os.getcwd(), '..')))

# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import torch
# from torch_geometric.utils import from_networkx
# import numpy as np
# from matplotlib import cm
# import torch.nn as nn
from generate_graph import GraphGeneration
from regular_tiling import gen_pyg_from_nx, plot_graph, plot_degree_histogram

if __name__ == "__main__":
    # square_grid: heterophily and homophily
    # triangle: homophily (heterophily doesn't exist)
    # hexagonal: heterophily and homophily
    #
    
    # Create a grid graph
    m, n = 4, 7

    for graph_type in ['triangle']:#'square_grid', 'triangle', 'hexagonal', 'kagome']:
        gen_graph = GraphGeneration(
            m=m,
            n=n,
            emb_dim=32,
            graph_type=graph_type,
            heterophily=False,
            homophily=True,
            feature_type='degree'
        )
        data, adj_matrix, node_features, pos = gen_graph.generate_graph()
        # for key, value in data.items():
        #     print(f"{key}: {type(value)}")
        print(data, '\n')
        print(data.pos, '\n')
        print(adj_matrix, '\n')
        # print(node_features, '\n')
        
        break


        # print(data.x)
        # print(data.edge_index)