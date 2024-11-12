import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join( os.getcwd(), '..')))

import argparse
from generate_graph import SyntheticGraphGeneration


# TODO: Argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Test SyntheticGraphGeneration with argparse.")
    parser.add_argument("--m", type=int, default=4, help="Number of rows (m) in the grid.")
    parser.add_argument("--n", type=int, default=4, help="Number of columns (n) in the grid.")
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--graph_type", type=str, default=0, choices=[0, 1, 2], help="Type of graph structure: Square: 0, Triangle: 1, Hexagonal: 2.")
    parser.add_argument("--heterophily", type=bool, default=False , help="Enable heterophily for the graph.")
    parser.add_argument("--homophily", type=bool, default=False , help="Enable heterophily for the graph.")
    parser.add_argument("--feature_type", type=str, default='random', choices=['random', 'one-hot', 'degree'], help="Embedding generation type.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    gen_graph = SyntheticGraphGeneration(
        m=args.m,
        n=args.n,
        emb_dim=args.emb_dim,
        graph_type=args.graph_type,
        heterophily=args.heterophily,
        homophily=args.homophily,
        feature_type=args.feature_type
    )
    data, adj_matrix, node_features, pos = gen_graph.generate_graph()
    
    # print(data, '\n')
    # print(data.pos, '\n')
    # print(adj_matrix, '\n')
    