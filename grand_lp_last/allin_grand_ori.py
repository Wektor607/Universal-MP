import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import time
from tqdm import tqdm
from torch_geometric.utils import (negative_sampling,
                                   to_undirected)
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 

from baselines.gnn_utils import  evaluate_hits, evaluate_auc, Logger, init_seed
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score
from data_utils.load_lp import randomsplit 
import torch_geometric.transforms as T
# from data_utils.load_lp import get_dataset
from data_utils.graph_rewiring import apply_KNN
from metrics.metrics import *
from models.base_classes import LinkPredictor
from models.GNN_KNN import GNN_KNN
from models.GNN_KNN_early import GNNKNNEarly
from models.GNN import GNN
from models.GNN_early import GNNEarly
from formatted_best_params import best_params_dict
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
from data_utils.graph_rewiring import rewire, apply_beltrami
import numpy as np


server = 'SDIL'

def get_dataset(root: str, opt: dict, name: str, use_valedges_as_input: bool=False, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(root="dataset", name=name)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
        try:
            data.num_nodes = data.x.shape[0]
        except:
            if hasattr(data, "num_nodes"):
                print(f"use default data.num_nodes: {data.num_nodes}.")
        
    data.edge_weight = None 
    data.adj_t = SparseTensor.from_edge_index(edge_index, 
                    sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # if name == "ogbl-collab":
    #     data.edge_weight = data.edge_weight/2
        
    if name == "ogbl-ppa":
        data.x = torch.argmax(data.x, dim=-1).unsqueeze(-1).float()
        data.max_x = torch.max(data.x).item()
    elif name == "ogbl-ddi":
        data.x = torch.arange(data.num_nodes).unsqueeze(-1).float()
        data.max_x = data.max_x = -1 # data.num_nodes
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



def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def visualize_predictions(pos_train_pred, neg_train_pred, 
                          pos_valid_pred, neg_valid_pred, 
                          pos_test_pred, neg_test_pred):
  
    """
    Visualizes the distribution of positive and negative predictions for train, validation, and test sets.
    """
    plt.figure(figsize=(15, 5))
    
    datasets = [(pos_train_pred, neg_train_pred, 'Train'),
                (pos_valid_pred, neg_valid_pred, 'Validation'),
                (pos_test_pred, neg_test_pred, 'Test')]
    
    for i, (pos_pred, neg_pred, title) in enumerate(datasets):
        plt.subplot(1, 3, i + 1)
        sns.histplot(pos_pred, bins=50, kde=True, color='#7FCA85', stat='density', label='Positive')
        sns.histplot(neg_pred, bins=50, kde=True, color='#BDAED2', stat='density', label='Negative', alpha=0.6)
        plt.title(f'{title} Set')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('visual_grand.png')


def plot_test_sequences(test_pred, test_true):
    """
    Plots test_pred as a line plot with transparent circles for positive and negative samples.
    """
    test_pred = test_pred.detach().cpu().numpy()  
    test_true = test_true.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(test_pred, marker='o', linestyle='-', label="Prediction Score", color='#7FCA85', alpha=0.7)
    plt.plot(test_true, marker='o', linestyle='-', label="True Score", color='#BDAED2', alpha=0.7)
    # Color true labels (1=Positive, 0=Negative)

    plt.xlabel("Sample Index")
    plt.ylabel("Prediction Score")
    plt.title("Test Predictions with True Labels")
    plt.legend()
    plt.savefig('plot_prediction.png')
    

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, mrr_mode=False, negative_data=None):
    preds = []
    if mrr_mode:
        source = input_data.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            preds += [score_func(h[src], h[dst_neg]).squeeze().cpu()]
        pred_all = torch.cat(preds, dim=0).view(-1, 1000)
    else:
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
        pred_all = torch.cat(preds, dim=0)
    return pred_all

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)
    
    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    return result
  

@torch.no_grad()
def test_epoch(opt, model, score_func, data, pos_encoding, batch_size, evaluation_edges, emb, evaluator_hit, evaluator_mrr, use_valedges_as_input):
    model.eval()
    predictor.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    if emb == None: x = data.x
    else: x = emb.weight
    
    if opt['gcn']:
      h = model(x, data.adj_t)
    else:
      h = model(data.x, pos_encoding)
    x1 = h
    x2 = torch.tensor(1)

    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.full_adj_t.to(x.device))
        x2 = h
        
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size)
    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)
    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.full_adj_t.to(x.device))
        x2 = h
    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)
    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size)
    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    return result, score_emb
  
    
def train_epoch(opt, predictor, model, optimizer, data, pos_encoding, splits, batch_size):
    predictor.train()
    model.train()
    
    pos_encoding = pos_encoding.to(model.device) if pos_encoding is not None else None
    pos_train_edge = splits['train']['edge'].to(data.x.device)
    neg_train_edge = negative_sampling(
        data.edge_index.to(pos_train_edge.device),
        num_nodes=data.num_nodes,
        num_neg_samples=pos_train_edge.size(0)
    ).t().to(data.x.device)

    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
        
    total_loss = total_examples = 0
    # DEBUG change it into DataLoader
    indices = torch.randperm(pos_train_edge.size(0), device=pos_train_edge.device)

    for start in tqdm(range(0, pos_train_edge.size(0), batch_size)):

        optimizer.zero_grad()
        if opt['gcn']:
          h = model(x, data.adj_t)
        else:
          h = model(data.x, pos_encoding)
        
        end = start + batch_size
        perm = indices[start:end]
        pos_out = predictor(h[pos_train_edge[perm, 0]], h[pos_train_edge[perm, 1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_out = predictor(h[neg_train_edge[perm, 0]], h[neg_train_edge[perm, 1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        if (not opt['gcn']) and (model.odeblock.nreg > 0):
            reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
            regularization_coeffs = model.regularization_coeffs
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss
        
        num_examples = (end - start)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
        if not opt['gcn']:
          model.fm.update(model.getNFE())
          model.resetNFE()
        
        print("=== Named parameters of the ODEFunc ===")
        for name, param in model.odeblock.odefunc.named_parameters():
            print(name, param.shape, param.requires_grad)
        raise(0)
        print("GRAD before backward:")  
        for name, param in model.odeblock.odefunc.named_parameters():
            print(f"{name}: {param.grad}")

        loss.backward()

        # После backward:
        print("GRAD after backward:")
        for name, param in model.odeblock.odefunc.named_parameters():
            if param.grad is None:
                print(f"❌ No grad for {name}!")
            else:
                print(f"✅ {name}, grad mean={param.grad.mean().item():.4f}, std={param.grad.std().item():.4f}")
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()
        
        if not opt['gcn']:
          model.bm.update(model.getNFE())
          model.resetNFE()
        #######################
        
    return total_loss / total_examples

def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = False
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  return opt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)

import random
def set_seed(seed=42):
    random.seed(seed)  # Фиксируем seed для стандартного random
    np.random.seed(seed)  # Фиксируем seed для NumPy
    torch.manual_seed(seed)  # Фиксируем seed для PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Фиксируем seed для PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Фиксируем seed для всех GPU
    torch.backends.cudnn.deterministic = True  # Опционально: делаем вычисления детерминированными
    torch.backends.cudnn.benchmark = False  # Отключаем автооптимизации для детерминированности

set_seed(999)
      
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='yamls/cora/gcn.yaml',
                        help='The configuration file path.')
    ### MPLP PARAMETERS ###
    # dataset setting
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    parser.add_argument('--print_summary', type=str, default='')

    ### GRAND PARAMETERS ###
    parser.add_argument('--use_cora_defaults', action='store_true',
                    help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--data_norm', type=str, default='rw',
                    help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                    help='use the 10 fixed splits from '
                        'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                    help='the number of splits to repeat the results on')
    parser.add_argument('--label_rate', type=float, default=0.5,
                    help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                    help='use planetoid splits for Cora/Citeseer/Pubmed')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                    help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                    help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                    help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                    help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                    help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                    help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                    help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                    help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                    help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                    help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                    help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                    help="Maximum number steps for the dopri5Early test integrator. "
                        "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                    help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                    help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                    help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                    help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                    help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    # topk is not implemented in the original implementation
    parser.add_argument('--gdc_sparsification', type=str, default='threshold', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                    help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                    help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                    help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                    help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                    help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")

    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                    help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                    help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                    help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    parser.add_argument('--symmetric_attention', action='store_true',
                    help='maks the attention symmetric for rewring in QK space')

    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    # MY PARAMETERS
    parser.add_argument('--mlp_num_layers', type=int, default=3, help="Number of layers in MLP")
    parser.add_argument('--batch_size', type=int, default=2**12)
    
    # gcn
    parser.add_argument('--gcn', type=str2bool, default=False)
    parser.add_argument('--num_layers', type=int, default=3)
    
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    args = parser.parse_args()

    cmd_opt = vars(args)
    try:
      best_opt = best_params_dict[cmd_opt['data_name']]
      opt = {**cmd_opt, **best_opt}
      merge_cmd_args(cmd_opt, opt)
    except KeyError:
      opt = cmd_opt
    args.name_tag = f"{args.data_name}_beltrami{args.beltrami}_mlp_score_epochs{args.epoch}_runs{args.runs}"
    
    opt['beltrami'] = False
    if opt['gcn']:
      opt['hidden_dim'] = 256
      opt['lr'] = 0.001
      opt['batch_size'] = 16394
    

    init_seed(999)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    data, splits = get_dataset(opt['dataset_dir'], opt, opt['data_name'], opt['use_valedges_as_input'])
    edge_index = data.edge_index
    print(data)
    emb = None
    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            data.x = data.x.to(torch.float)
            data.x = data.x.to(device)
            input_channel = data.x.size(1)
        else:
            emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
            input_channel = args.hidden_channels
    else:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
        
    if args.data_name == "ogbl-citation2":
        opt['metric'] = "MRR"
    if data.x is None:
        opt['use_feature'] = False
    
    if opt['beltrami']:
      print("Applying Beltrami")
      pos_encoding = apply_beltrami(data.to('cpu'), opt).to(device)
      opt['pos_enc_dim'] = pos_encoding.shape[1]
      print(f"pos encoding is {pos_encoding}")
      print(f"pos encoding shape is {pos_encoding.shape}")
      print(f"pos encoding type is {type(pos_encoding)}")
    else:
      pos_encoding = None

    print(data, args.data_name)
    if data.edge_weight is None:
        edge_weight = torch.ones((data.edge_index.size(1), 1))
        print(f"custom edge_weight {edge_weight.size()} added for {args.data_name}")
    data = T.ToSparseTensor()(data)
    data.edge_index = edge_index
    if args.use_valedges_as_input:
        val_edge_index = splits['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'mrr_hit1':  Logger(args.runs),
        'mrr_hit3':  Logger(args.runs),
        'mrr_hit10':  Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'
    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'
    elif args.data_name in ['Cora', 'Pubmed', 'Citeseer']:
        eval_metric = 'Hits@100'

    if args.data_name != 'ogbl-citation2':
        pos_train_edge = splits['train']['edge']
        pos_valid_edge = splits['valid']['edge']
        neg_valid_edge = splits['valid']['edge_neg']
        pos_test_edge = splits['test']['edge']
        neg_test_edge = splits['test']['edge_neg']
        
    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2
    
    data = data.to(device)
    predictor = LinkPredictor(opt['hidden_dim'], opt['hidden_dim'], 1, opt['mlp_num_layers'], opt['dropout']).to(device)
    batch_size = opt['batch_size']  
    if opt['gcn']:
        model = GCN(data.x.shape[1],  opt['hidden_dim'], opt['hidden_dim'], opt['num_layers'], opt['dropout']).to(device)
    else:
      if opt['rewire_KNN'] or opt['fa_layer']:
        model = GNN_KNN(opt, data, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, data, splits, predictor, batch_size, device).to(device)
      else:
        print(opt["no_early"])
        model = GNN(opt, data, device).to(device) if opt["no_early"] else GNNEarly(opt, data, splits, predictor, batch_size, device).to(device)

    parameters = (
      [p for p in model.parameters() if p.requires_grad] +
      [p for p in predictor.parameters() if p.requires_grad]
    )
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])

    best_epoch = 0
    best_metric = 0
    best_results = None

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]

    for run in range(args.runs):
      print('#################################          ', run, '          #################################')
      # import wandb
      if opt['gcn']:
        name_tag = f"{args.data_name}_gcn_{server}_{args.runs}"
      else:
        name_tag = f"{args.data_name}_grand_{server}_{args.runs}"
      # wandb.init(project="GRAND4LP", name=name_tag, config=opt)
      if args.runs == 1:
          seed = 0
      else:
          seed = run
      print('seed: ', seed)
      init_seed(seed)
      model.reset_parameters()
      predictor.reset_parameters()

      best_valid = 0
      kill_cnt = 0
      best_test = 0
      step = 0
        
      print(f"Starting training for {opt['epoch']} epochs...")
      for epoch in tqdm(range(1, opt['epoch'] + 1)):
          start_time = time.time()

          # CHECK Misalignment
          # if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
          #     ei = apply_KNN(data, pos_encoding, model, opt)
          #     model.odeblock.odefunc.edge_index = ei
                 
          loss = train_epoch(opt, predictor, model, optimizer, data, pos_encoding, splits, batch_size)
          print(f"Epoch {epoch}, Loss: {loss:.4f}")
          # wandb.log({'train_loss': loss}, step = epoch)
          step += 1
          
          if epoch % args.eval_steps == 0:
              results, score_emb = test_epoch(opt, model, predictor, data, pos_encoding, batch_size, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.use_valedges_as_input)
              
              for key, result in results.items():
                  loggers[key].add_result(run, result)
                  # wandb.log({f"Metrics/{key}": result[-1]}, step=step)
                  
              current_metric = results['Hits@100'][2]
              if current_metric > best_metric:
                  best_epoch = epoch
                  best_metric = current_metric
                  best_results = results
              print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
              print(f"Current Best {current_metric}: {best_metric:.4f} (Epoch {best_epoch})")
      print(f"Training completed. Best {current_metric}: {best_metric:.4f} (Epoch {best_epoch})")

      for key in loggers.keys():
          if len(loggers[key].results[0]) > 0:
              print(key)
              loggers[key].print_statistics(run)

    result_all_run = {}
    save_dict = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
    print(f"now save {save_dict}")
    print(f"to results_ogb_gnn/{args.data_name}_lm_mrr.csv")
    print(f"with name {args.name_tag}.")
    mvari_str2csv(args.name_tag, save_dict, f'results_grand_gnn/{args.data_name}_lm_mrr.csv')

    print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))
