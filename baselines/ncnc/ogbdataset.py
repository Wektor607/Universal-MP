import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import  is_undirected

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge


import torch
from torch_sparse import SparseTensor

def is_symmetric(adj_t: SparseTensor) -> bool:
    # Checks whether a given SparseTensor is symmetric.
    return (adj_t.t() == adj_t)


# TODO document the standard preprocessing of dataset, categoried by data name, 
# TODO node feature preprocessing, resource, visualization
# TODO split dataset visualization 
# TODO edge weight visualization
# TODO merge loaddataset with get_dataset to simplify the comparison
def loaddataset(name: str, use_valedges_as_input: bool, load=None):
        
    if name in ['ppa', 'ddi', 'collab', 'citation2', 'vessel']:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        edge_index = data.edge_index
    elif name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        raise ValueError(f"Dataset {name} not supported")

    # copy from get_dataset
    if 'edge_weight' in data: 
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        print(f"{name}: edge_weight max: {data.edge_weight.max()}")
    else:
        data.edge_weight = None 
        print(f"{name}: edge_weight not found")
    
    # symmetric and coalesce adj 
    print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
                
    print(f"is symmetric {is_symmetric(data.adj_t)}")
    print(f"is undirected {is_undirected(data.edge_index)}")
    
    print(data.x)
    data.max_x = -1
    if name == "ppa":
        # transform one-hot to scalar
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        # ddi no node feature
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    # 
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # Use training + valid edges 
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge


if __name__ == "__main__":

    data, split_edge = loaddataset("collab", False)
    data, split_edge = loaddataset("citation2", False)
    data, split_edge = loaddataset("ddi", False)
    data, split_edge = loaddataset("vessel", False)
    data, split_edge = loaddataset("ppa", False)
        
    data, split_edge = loaddataset("Cora", False)
    data, split_edge = loaddataset("Citeseer", False)
    data, split_edge = loaddataset("Pubmed", False)
    