import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import torch_geometric.transforms as T

def sort_edge_index(edge_index):
    """Sort the edge index in ascending order according to the source node index."""

    src_index, sort_indices = torch.sort(edge_index[:, 0])
    dst_index = edge_index[sort_indices, 1]
    edge_reindex = torch.stack([src_index, dst_index])
    return edge_reindex, sort_indices


def randomsplit(data, val_ratio: float = 0.05, test_ratio: float = 0.15):
    data.edge_index, _ = coalesce(
        data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if not hasattr(data, 'x'):
        data.num_nodes = data.x.shape[0]

    transform = T.Compose([
            T.NormalizeFeatures(),
            RandomLinkSplit(
                is_undirected=True,
                num_val=val_ratio,
                num_test=test_ratio,
                add_negative_train_samples=True,
                split_labels=True
            )
        ])
    train_data, val_data, test_data = transform(data)
    splits = {'train': {}, 'valid': {}, 'test': {}}
    splits['train'] = train_data
    splits['valid'] = val_data
    splits['test'] = test_data
    for split_name, split in splits.items():
        split['x'] = data.x
    del data, train_data, val_data, test_data
    return splits


def loaddataset(name: str, nfeat_path=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        data = dataset[0]
        splits = randomsplit(data)
        data.edge_index = to_undirected(splits["train"]["pos_edge_label_index"])
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        data = dataset[0]
        if name == "ppa":
            data.x = torch.argmax(data.x, dim=-1).float()
            data.max_x = torch.max(data.x).item()
        elif name == "ddi":
            data.x = torch.diag(torch.arange(data.num_nodes).float())
            data.max_x = data.num_nodes
        splits = randomsplit(data)
        edge_index = data.edge_index
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1

    if nfeat_path:
        # comment ? 
        data.x = torch.load(nfeat_path, map_location="cpu")
    data.full_adj_t = data.adj_t
    return data, splits


if __name__ == "__main__":
    loaddataset("ddi", False)