import pytest
import torch
from generate_graph import SyntheticGraphGeneration

@pytest.mark.parametrize("graph_type", [0, 1, 2])
def test_generate_graph_basic(graph_type):
    gen = SyntheticGraphGeneration(m=5, n=5, emb_dim=8, graph_type=graph_type)
    data, adj_t, nodefeats, _ = gen.generate_graph()
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert nodefeats.shape[1] == 8
    assert adj_t.nnz() > 0

@pytest.mark.parametrize("m, n", [(3, 3), (5, 10), (10, 5)])
def test_generate_graph_with_different_number_of_rows_and_columns(m, n):
    gen = SyntheticGraphGeneration(m=m, n=n, emb_dim=4, graph_type=0)
    data, _, nodefeats, _ = gen.generate_graph()
    assert data.num_nodes == m * n
    assert nodefeats.shape[0] == m * n

@pytest.mark.parametrize("graph_type, max_neighbors", [
    (0, 4),
    (2, 3)
])
@pytest.mark.parametrize("m, n", [
    (4, 4),
    (4, 5),
    (4, 6)
])
def test_generate_graph_with_heterophily(graph_type, max_neighbors, m, n):
    gen = SyntheticGraphGeneration(m=m, n=n, emb_dim=6, graph_type=graph_type, heterophily=True)
    data, _, _, _ = gen.generate_graph()
    labels = data.y.numpy()
    
    edge_index = data.edge_index.numpy()
    
    neighbors = {i: set() for i in range(data.num_nodes)}
    for i in range(edge_index.shape[1]):
        node1 = edge_index[0, i]
        node2 = edge_index[1, i]
        neighbors[node1].add(node2)
        neighbors[node2].add(node1)

    for node, node_neighbors in neighbors.items():
        node_label = labels[node]
        for neighbor in node_neighbors:
            neighbor_label = labels[neighbor]
            assert node_label != neighbor_label, (
                f"Node {node} and its neighbor {neighbor} have the same label "
                f"in heterophilic graph of type {graph_type}."
            )
        
        # Check that the number of neighbors does not exceed the maximum value for the given graph type
        assert len(node_neighbors) <= max_neighbors, (
            f"Node {node} has {len(node_neighbors)} neighbors, which exceeds the "
            f"maximum of {max_neighbors} for graph type {graph_type}."
        )

def test_generate_graph_with_homophily():
    gen = SyntheticGraphGeneration(m=4, n=4, emb_dim=6, graph_type=2, homophily=True)
    data, _, _, _ = gen.generate_graph()
    labels = data.y.numpy()
    
    num_labels_0 = (labels == 0).sum()
    num_labels_1 = (labels == 1).sum()
    
    assert num_labels_0 != len(labels) // 2
    assert num_labels_1 != len(labels) // 2

def test_edge_weights():
    gen = SyntheticGraphGeneration(m=3, n=3, emb_dim=4, graph_type=1)
    data, _, _, _ = gen.generate_graph()
    weights = data.edge_weight.numpy()
    assert (weights >= 0.1).all() and (weights <= 1.0).all()

@pytest.mark.parametrize("feature_type", ["random", "one-hot", "degree"])
def test_nodefeat_generation(feature_type):
    gen = SyntheticGraphGeneration(m=5, n=5, emb_dim=4, graph_type=0, feature_type=feature_type)
    G, _ = gen.create_square_grid()
    features = gen.generate_nodefeats(G)
    if feature_type == "random":
        assert features.shape == (25, 4)
    elif feature_type == "one-hot":
        assert features.shape == (25, 25)
    elif feature_type == "degree":
        assert features.shape == (25, 1)
        degrees = [val for _, val in G.degree()]
        assert (features.squeeze().numpy() == degrees).all()
