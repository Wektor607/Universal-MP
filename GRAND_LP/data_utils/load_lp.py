import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch
import numpy as np
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
from torch_geometric import datasets
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, index_to_mask,
                                   is_undirected, negative_sampling,
                                   to_undirected, train_test_split_edges, coalesce)
from torch_geometric.transforms import (BaseTransform, Compose, ToSparseTensor,
                                        NormalizeFeatures, RandomLinkSplit,
                                        ToDevice, ToUndirected)
from torch_geometric.data.collate import collate
import scipy.io as sio
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms.two_hop import TwoHop
from metrics.distances_kNN import apply_dist_KNN, apply_dist_threshold, get_distances, apply_feat_KNN
from metrics.hyperbolic_distances import hyperbolize
from torch_geometric.transforms import GDC
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj, \
  dense_to_sparse, to_undirected
import torch_geometric.transforms as T
from torch_geometric.nn import Node2Vec
from typing import Union
import ast
from data_utils.lcc import *

from torch_geometric.datasets import Planetoid

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CLUSTER_FILENAME = f"{ROOT_DIR}/data/ddi_features/clustering.txt"
PAGERANK_FILENAME = f"{ROOT_DIR}/data/ddi_features/pagerank.txt"
DEGREE_FILENAME = f"{ROOT_DIR}/data/ddi_features/degree.pkl"
CENTRALITY_FILENAME = f"{ROOT_DIR}/data/ddi_features/centrality.pkl"

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
    split_edge['train']['edge'] = removerepeated(
        torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def get_dataset(root: str, opt: dict, name: str, use_valedges_as_input: bool, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=name)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
    data.edge_weight = None 
    
    data.adj_t = SparseTensor.from_edge_index(edge_index, 
                    sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ogbl-ppa":
        data.x = torch.argmax(data.x, dim=-1).unsqueeze(-1).float()
        data.max_x = torch.max(data.x).item()
    elif name == "ogbl-ddi":
        data.x = torch.arange(data.num_nodes).unsqueeze(-1).float()
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1
    
    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                            sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
        if opt['rewiring'] is not None:
            data.edge_index = full_edge_index.copy()
            data = rewire(data, opt, root)
    else:
        data.full_adj_t = data.adj_t
        if opt['rewiring'] is not None:
            data = rewire(data, opt, root)
    return data, split_edge
  
# def get_dataset(root, name: str, opt: dict, use_valedges_as_input=False, device='cpu', year=-1):
#     if name.startswith('ogbl-'):
#         dataset = PygLinkPropPredDataset(name=name, root=root)

#         data = dataset[0]

#         # Idea from original repo: https://github.com/chuanqichen/cs224w/blob/main/ddi/gnn_augmented_node2vec.py
#         if name == 'ogbl-ddi':
#           data.x = get_features(data.num_nodes)
#           emb = Node2Vec(data.edge_index, opt['hidden_dim'] - data.x.size(1), walk_length=20,
#                       context_size=10, walks_per_node=10)
#           embeddings = emb()

#           data.x = torch.cat([data.x.to(device), embeddings.to(device)], dim=1)
#           dataset.data = data
#         else:
#           emb = None
        
#         if data.x is not None:
#           data.x = data.x.to(torch.float)
#         # TODO: CHECK (VERY LONGG)
#         # if opt['use_lcc']:
#         #   data, lcc, _ = use_lcc(data)
#         #   dataset.data = data
#           # print(data.x.shape, data.edge_index.shape)
#           # lcc = get_largest_connected_component(dataset)

#           # data.x = data.x[lcc]

#           # row, col = dataset.data.edge_index.numpy()
#           # edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
#           # data.edge_index = torch.tensor(remap_edges(edges, get_node_mapper(lcc)))
#           # print(data.x.shape, data.edge_index.shape)
        
#         split_edge = dataset.get_edge_split()

#         if name == 'ogbl-collab' and year > 0:
#             data, split_edge = filter_by_year(data, split_edge, year)
        
#         if name == 'ogbl-vessel':
#             data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
#             data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
#             data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
          
#         if 'edge_weight' in data:
#             data.edge_weight = data.edge_weight.view(-1).to(torch.float)
#             if name == "ogbl-collab":
#                 data.edge_weight = data.edge_weight / 2

#         data = ToSparseTensor(remove_edge_index=False)(data)
#         data.adj_t = data.adj_t.to_symmetric()

#         if use_valedges_as_input:
#             val_edge_index = split_edge['valid']['edge_index'].t()
#             full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
#             data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
#                                                     sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
#             data.full_adj_t = data.full_adj_t.to_symmetric()
#             if opt['rewiring'] is not None:
#                 data.edge_index = full_edge_index.copy()
#                 data = rewire(data, opt, root)
#         else:
#             data.full_adj_t = data.adj_t
#             if opt['rewiring'] is not None:
#                 data = rewire(data, opt, root)
        
#         # if name != 'ogbl-ddi':    
#         #   data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
#         print(data)
#         return data, split_edge, emb

#     pyg_dataset_dict = {
#         'Cora': (datasets.Planetoid, {'name':'Cora'}),
#         'Citeseer': (datasets.Planetoid, {'name':'Citeseer'}),
#         'Pubmed': (datasets.Planetoid, {'name':'Pubmed'}),
#         'CS': (datasets.Coauthor, {'name':'CS'}),
#         'Physics': (datasets.Coauthor, {'name':'physics'}),
#         'Computers': (datasets.Amazon, {'name':'Computers'}),
#         'Photo': (datasets.Amazon, {'name':'Photo'}),
#         'PolBlogs': (datasets.PolBlogs, {}),
#     }

#     if name in pyg_dataset_dict:
#         dataset_class, kwargs = pyg_dataset_dict[name]
#         dataset = dataset_class(root=root, transform=ToUndirected(), **kwargs)
#         data, _, _ = collate(
#                 dataset[0].__class__,
#                 data_list=list(dataset),
#                 increment=True,
#                 add_batch=False,
#             )
#         undirected = data.is_undirected()
#         split_edge = get_edge_split(data,
#                    undirected,
#                    device,
#                    0.15,
#                    0.05,
#                    True,
#                    True)
#         data.adj_t = SparseTensor.from_edge_index(data.edge_index, 
#                                                   sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
#     else:
#         data = load_unsplitted_data(root, name)
    
#     if opt['rewiring'] is not None:
#         data = rewire(data, opt, root)
    
#     return data, split_edge, None

def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper

def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes

from collections import Counter
# For ogbl-ddi
def get_features(n_nodes):
    with open(PAGERANK_FILENAME, "r") as f:
        contents = f.read()
        pagerank_dict = ast.literal_eval(contents)
    pagerank_vals = torch.FloatTensor(list(pagerank_dict.values())).reshape((n_nodes, 1))
    
    # pagerank_vals = (pagerank_vals - pagerank_vals.mean()) / pagerank_vals.std() 
    
    with open(CLUSTER_FILENAME, "r") as f:
        contents = f.read()
        clustering_dict = ast.literal_eval(contents)
    cluster_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))
    
    # cluster_vals = (cluster_vals - cluster_vals.mean()) / cluster_vals.std()
    # # ERROR
    # with open(DEGREE_FILENAME, "rb") as f:
    #     degree_dict = pickle.load(f)
    # print(list(degree_dict.values()))
    # # degree_vals = torch.FloatTensor(list(degree_dict.values())).reshape((n_nodes, 1))
    # # My:
    # degree_vals = torch.FloatTensor([
    #   sum(counter.values()) / len(counter.values()) if isinstance(counter, Counter) and len(counter) > 0 else 0
    #   for counter in degree_dict.values()
    # ]).reshape((n_nodes, 1))

    # print(degree_vals)
    # raise(0)
    # with open(CENTRALITY_FILENAME, "rb") as f:
    #     centrality_dict = pickle.load(f)
    # # centrality_vals = torch.FloatTensor(list(centrality_dict.values())).reshape((n_nodes, 1))
    # centrality_vals = [
    #     sum(v.values()) / len(v) if isinstance(v, dict) and len(v) > 0 else 0
    #     for v in centrality_dict.values()
    # ]
    # print(centrality_vals)
    # raise(0)
    ones = torch.ones((n_nodes, 1))
    # features = torch.cat((ones, pagerank_vals, cluster_vals, centrality_vals, degree_vals), 1)
    features = torch.cat((ones, pagerank_vals, cluster_vals, cluster_vals, cluster_vals), 1)
    return features

def get_edge_split(data: Data,
                   undirected: bool,
                   device: Union[str, int],
                   val_pct: float,
                   test_pct: float,
                   include_negatives: bool,
                   split_labels: bool):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),

    ])
    del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
    print(data)

    train_data, val_data, test_data = transform(data)
    return {'train': train_data, 'valid': val_data, 'test': test_data}

def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge

def data_summary(name: str, data: Data, header=False, latex=False):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    n_degree = data.adj_t.sum(dim=1).to(torch.float)
    avg_degree = n_degree.mean().item()
    degree_std = n_degree.std().item()
    max_degree = n_degree.max().long().item()
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    if data.x is not None:
        attr_dim = data.x.shape[1]
    else:
        attr_dim = '-' # no attribute

    if latex:
        latex_str = ""
        if header:
            latex_str += r"""
            \begin{table*}[ht]
            \begin{center}
            \resizebox{0.85\textwidth}{!}{
            \begin{tabular}{lccccccc}
                \toprule
                \textbf{Dataset} & \textbf{\#Nodes} & \textbf{\#Edges} & \textbf{Avg. node deg.} & \textbf{Std. node deg.} & \textbf{Max. node deg.} & \textbf{Density} & \textbf{Attr. Dimension}\\
                \midrule"""
        latex_str += f"""
                \\textbf{{{name}}}"""
        latex_str += f""" & {num_nodes} & {num_edges} & {avg_degree:.2f} & {degree_std:.2f} & {max_degree} & {density*100:.4f}\% & {attr_dim} \\\\"""
        latex_str += r"""
                \midrule"""
        if header:
            latex_str += r"""
            \bottomrule
            \end{tabular}
            }
            \end{center}
            \end{table*}"""
        print(latex_str)
    else:
        print("-"*30+'Dataset and Features'+"-"*60)
        print("{:<10}|{:<10}|{:<10}|{:<15}|{:<15}|{:<15}|{:<10}|{:<15}"\
            .format('Dataset','#Nodes','#Edges','Avg. node deg.','Std. node deg.','Max. node deg.', 'Density','Attr. Dimension'))
        print("-"*110)
        print("{:<10}|{:<10}|{:<10}|{:<15.2f}|{:<15.2f}|{:<15}|{:<9.4f}%|{:<15}"\
            .format(name, num_nodes, num_edges, avg_degree, degree_std, max_degree, density*100, attr_dim))
        print("-"*110)

def load_unsplitted_data(root,name):
    '''
    This function is used to load and prepare graph data in the format ".mat" 
    for later use in PyTorch Geometric library models
    '''
    # read .mat format files
    data_dir = root + '/{}.mat'.format(name)
    # print('Load data from: '+ data_dir)
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    return data

def rewire(data, opt, data_dir):
    rw = opt['rewiring']
    if rw == 'two_hop':
        data = get_two_hop(data)
    elif rw == 'gdc':
        data = apply_gdc(data, opt)
    # Didn't works
    elif rw == 'pos_enc_knn':
        data = apply_pos_dist_rewire(data, opt, data_dir)
    return data

def get_two_hop(data):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  th = TwoHop()
  data = th(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data

def apply_gdc(data, opt, type="combined"):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  print('performing gdc transformation with method {}, sparsification {}'.format(opt['gdc_method'],
                                                                                 opt['gdc_sparsification']))
  if opt['gdc_method'] == 'ppr':
    diff_args = dict(method='ppr', alpha=opt['ppr_alpha'])
  else:
    diff_args = dict(method='heat', t=opt['heat_time'])
  if opt['gdc_sparsification'] == 'topk':
    sparse_args = dict(method='topk', k=opt['gdc_k'], dim=0)
    diff_args['eps'] = opt['gdc_threshold']
  else:
    sparse_args = dict(method='threshold', eps=opt['gdc_threshold'])
    diff_args['eps'] = opt['gdc_threshold']
  print('gdc sparse args: {}'.format(sparse_args))
  if opt['self_loop_weight'] != 0:
    gdc = GDCWrapper(float(opt['self_loop_weight']),
                     normalization_in='sym',
                     normalization_out='col',
                     diffusion_kwargs=diff_args,
                     sparsification_kwargs=sparse_args, exact=opt['exact'])
  else:
    gdc = GDCWrapper(self_loop_weight=None,
                     normalization_in='sym',
                     normalization_out='col',
                     diffusion_kwargs=diff_args,
                     sparsification_kwargs=sparse_args, exact=opt['exact'])
  if isinstance(data.num_nodes, list):
    data.num_nodes = data.num_nodes[0]

  if type == 'combined':
    data = gdc(data)
  elif type == 'pos_encoding':
    if opt['pos_enc_orientation'] == "row":  # encode row of S_hat
      return gdc.position_encoding(data)
    elif opt['pos_enc_orientation'] == "col":  # encode col of S_hat
      return gdc.position_encoding(data).T

  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data

def apply_pos_dist_rewire(data, opt, data_dir='../data'):
  if opt['pos_enc_type'].startswith("HYP"):
    pos_enc_dir = os.path.join(f"{data_dir}", "pos_encodings")
    # generate new positional encodings distances
    # do encodings already exist on disk?
    fname = os.path.join(pos_enc_dir, f"{opt['dataset']}_{opt['pos_enc_type']}_dists.pkl")
    print(f"[i] Looking for positional encoding DISTANCES in {fname}...")

    # - if so, just load them
    if os.path.exists(fname):
      print("    Found them! Loading cached version")
      with open(fname, "rb") as f:
        pos_dist = pickle.load(f)
      # if opt['pos_enc_type'].startswith("DW"):
      #   pos_dist = pos_dist['data']

    # - otherwise, calculate...
    else:
      print("    Encodings not found! Calculating and caching them")
      # choose different functions for different positional encodings
      if opt['pos_enc_type'].startswith("HYP"):
        pos_encoding = apply_beltrami(data, opt)
        pos_dist = hyperbolize(pos_encoding)


      else:
        print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
        quit()
      # - ... and store them on disk
      POS_ENC_PATH = os.path.join(data_dir, "pos_encodings")
      if not os.path.exists(POS_ENC_PATH):
        os.makedirs(POS_ENC_PATH)

      # if opt['pos_enc_csv']:
      #   sp = pos_encoding.to_sparse()
      #   table_mat = np.concatenate([sp.indices(), np.atleast_2d(sp.values())], axis=0).T
      #   np.savetxt(f"{fname[:-4]}.csv", table_mat, delimiter=",")

      with open(fname, "wb") as f:
        pickle.dump(pos_dist, f)

      if opt['gdc_sparsification'] == 'topk':
        ei = apply_dist_KNN(pos_dist, opt['gdc_k'])
      elif opt['gdc_sparsification'] == 'threshold':
        ei = apply_dist_threshold(pos_dist, opt['pos_dist_quantile'])

  elif opt['pos_enc_type'].startswith("DW"):
    pos_encoding = apply_beltrami(data, opt, data_dir)
    if opt['gdc_sparsification'] == 'topk':
      ei = apply_feat_KNN(pos_encoding, opt['gdc_k'])
      # ei = KNN(pos_encoding, opt)
    elif opt['gdc_sparsification'] == 'threshold':
      dist = get_distances(pos_encoding)
      ei = apply_dist_threshold(dist)

  data.edge_index = torch.from_numpy(ei).type(torch.LongTensor)

  return data


class GDCWrapper(GDC):
  def __init__(self, self_loop_weight=1, normalization_in='sym',
               normalization_out='col',
               diffusion_kwargs=dict(method='ppr', alpha=0.15),
               sparsification_kwargs=dict(method='threshold',
                                          avg_degree=64), exact=True):
    super(GDCWrapper, self).__init__(self_loop_weight, normalization_in, normalization_out, diffusion_kwargs,
                              sparsification_kwargs, exact)
    self.self_loop_weight = self_loop_weight
    self.normalization_in = normalization_in
    self.normalization_out = normalization_out
    self.diffusion_kwargs = diffusion_kwargs
    self.sparsification_kwargs = sparsification_kwargs
    self.exact = exact

    if self_loop_weight:
      assert exact or self_loop_weight == 1

  def position_encoding(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                             **self.diffusion_kwargs)
      edge_index, edge_weight = dense_to_sparse(diff_mat)
      # edge_index, edge_weight = self.sparsify_dense(
      #   diff_mat, **self.sparsification_kwargs)
    else:
      edge_index, edge_weight = self.diffusion_matrix_approx(
        edge_index, edge_weight, N, self.normalization_in,
        **self.diffusion_kwargs)
      # edge_index, edge_weight = self.sparsify_sparse(
      #   edge_index, edge_weight, N, **self.sparsification_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    return to_dense_adj(edge_index,
                        edge_attr=edge_weight).squeeze()


def apply_beltrami(data, opt, data_dir=f'{ROOT_DIR}/data'):
  pos_enc_dir = os.path.join(f"{data_dir}", "pos_encodings")
  # generate new positional encodings
  # do encodings already exist on disk?
  fname = os.path.join(pos_enc_dir, f"{opt['dataset']}_{opt['pos_enc_type']}.pkl")
  print(f"[i] Looking for positional encodings in {fname}...")

  # - if so, just load them
  if os.path.exists(fname):
    print("    Found them! Loading cached version")
    with open(fname, "rb") as f:
      # pos_encoding = pickle.load(f)
      pos_encoding = pickle.load(f)
    if opt['pos_enc_type'].startswith("DW"):
      pos_encoding = pos_encoding['data']

  # - otherwise, calculate...
  else:
    print("    Encodings not found! Calculating and caching them")
    # choose different functions for different positional encodings
    if opt['pos_enc_type'] == "GDC":
      pos_encoding = apply_gdc(data, opt, type="pos_encoding")
    else:
      print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
      quit()
    # - ... and store them on disk
    POS_ENC_PATH = os.path.join(data_dir, "pos_encodings")
    if not os.path.exists(POS_ENC_PATH):
      os.makedirs(POS_ENC_PATH)

    if opt['pos_enc_csv']:
      sp = pos_encoding.to_sparse()
      table_mat = np.concatenate([sp.indices(), np.atleast_2d(sp.values())], axis=0).T
      np.savetxt(f"{fname[:-4]}.csv", table_mat, delimiter=",")

    with open(fname, "wb") as f:
      pickle.dump(pos_encoding, f)

  return pos_encoding
