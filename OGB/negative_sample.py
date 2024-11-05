# -*- coding: utf-8 -*-
import torch
from torch_geometric.utils import negative_sampling, add_self_loops


def global_neg_sample(edge_index, num_nodes, num_samples,
                      num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples * num_neg, method=method)

    neg_src = neg_edge[0]
    neg_dst = neg_edge[1]
    if neg_edge.size(1) < num_samples * num_neg:
        print(f"not enough neg samples..")
        k = num_samples * num_neg - neg_edge.size(1)
        rand_index = torch.randperm(neg_edge.size(1))[:k]
        neg_src = torch.cat((neg_src, neg_src[rand_index]))
        neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
    return torch.reshape(torch.stack((neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def global_perm_neg_sample(edge_index, num_nodes, num_samples,
                           num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples, method=method)
    return sample_perm_copy(neg_edge, num_samples, num_neg)


def local_neg_sample(pos_edges, num_nodes, num_neg, random_src=False):
    """
    Generates negative samples based on the given positive edges. The negative samples 
    are created by either randomly selecting a source node from the positive edges or 
    using the same source nodes as in the positive edges, and then pairing them with 
    randomly selected destination nodes.

    Parameters:
    pos_edges (torch.Tensor): A tensor of shape (num_edges, 2) representing the positive edges. 
                              Each row corresponds to an edge with a source and destination node.
    num_nodes (int): The total number of nodes in the graph. Used to sample random destination nodes.
    num_neg (int): The number of negative edges to sample per positive edge.
    random_src (bool): If True, randomly selects the source node from either the source or destination 
                       of the positive edges. If False, uses the source node from the positive edge.

    Returns:
    torch.Tensor: A tensor of shape (num_edges, num_neg, 2), where each row represents a set of 
                  negative edges corresponding to each positive edge.
    """
    # TODO this method could sample pos edges 
    if random_src:
        # choose source 
        neg_src = pos_edges[torch.arange(pos_edges.size(0)), torch.randint(0, 2, (pos_edges.size(0),), dtype=torch.long)]
    else:
        neg_src = pos_edges[:, 0]
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst = torch.randint(0, num_nodes, (num_neg * pos_edges.size(0),), dtype=torch.long)

    return torch.reshape(torch.stack((neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


# def local_dist_neg_sample(pos_edges, num_neg, neg_table, random_src=True):
#     if random_src:
#         neg_src = pos_edges[torch.arange(pos_edges.size(0)), torch.randint(
#             0, 2, (pos_edges.size(0),), dtype=torch.long)]
#     else:
#         neg_src = pos_edges[:, 0]
#     neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
#     neg_src = torch.reshape(neg_src, (-1,))
#     neg_dst_index = torch.randint(
#         0, neg_table.size(0), (num_neg * pos_edges.size(0),), dtype=torch.long)
#     neg_dst = neg_table[neg_dst_index]
#     return torch.reshape(torch.stack(
#         (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def sample_perm_copy(edge_index, target_num_sample, num_perm_copy):
    src = edge_index[0]
    dst = edge_index[1]
    if edge_index.size(1) < target_num_sample:
        k = target_num_sample - edge_index.size(1)
        rand_index = torch.randperm(edge_index.size(1))[:k]
        src = torch.cat((src, src[rand_index]))
        dst = torch.cat((dst, dst[rand_index]))
    tmp_src = src
    tmp_dst = dst
    for i in range(num_perm_copy - 1):
        rand_index = torch.randperm(target_num_sample)
        src = torch.cat((src, tmp_src[rand_index]))
        dst = torch.cat((dst, tmp_dst[rand_index]))
    return torch.reshape(torch.stack(
        (src, dst), dim=-1), (-1, num_perm_copy, 2))

