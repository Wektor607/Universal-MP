
import os
import sys
import csv
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time
from yacs.config import CfgNode
from pprint import pprint 
import numpy as np
import random

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from baselines.heuristic import AA, RA, Ben_PPR, katz_apro, shortest_path
from baselines.heuristic import CN as CommonNeighbor
from baselines.GNN import LinkPredictor
from yacs.config import CfgNode as CN
from baselines.LINKX import LINKX
from baselines.GNN import Custom_GAT, Custom_GCN, GraphSAGE, Custom_GIN
from trials.data_utils import loaddataset
from utils import save_to_csv, visualize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm 
from utils import EarlyStopping
import argparse
import wandb
from typing import Dict, Any
import pickle
from baselines.HLGNN import HLGNN


ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"ROOT: {ROOT}")
NUM_SEED = 8

# Training parameters 

class Config_ddi_CN:
    def __init__(self):
        self.epochs = 1
        self.dataset = "ddi"
        self.batch_size = 2**10
        self.heuristic = "CN" # convergenced for and "CN"
        self.model = "GIN_Variant" #  "GIN_Variant", "GAT", "GCN", "GraphSAGE", "LINKX"
        Nodefeat = 'adjacency'  # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True
        self.lr = 0.001
        

class Config_ddi_PPR:
    def __init__(self):
        self.epochs = 1
        self.dataset = "ddi"
        self.batch_size = 2**10
        self.heuristic = "PPR" # convergenced for and "CN"
        self.model = "GIN_Variant" #  "GIN_Variant", "GAT", "GCN", "GraphSAGE", "LINKX"
        Nodefeat = 'adjacency'  # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True
        self.lr = 0.001
        
        
class Config_Cora_CN:
    def __init__(self):
        self.epochs = 4
        self.dataset = "Cora"
        self.batch_size = 2**4
        self.heuristic = "CN" # convergenced for and "CN"
        self.model = "GIN_Variant" #  "GIN_Variant", "GAT", "GCN", "GraphSAGE", "LINKX"
        Nodefeat = 'adjacency'  # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True
        self.ls = 0.04


class Config_Cora_PPR:
    def __init__(self):
        self.epochs = 4
        self.dataset = "Cora"
        self.batch_size = 2**4
        self.heuristic = "PPR" # convergenced for and "CN"
        self.model = "GIN_Variant" #  "GIN_Variant", "GAT", "GCN", "GraphSAGE", "LINKX"
        Nodefeat = 'adjacency'  # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True
        self.ls = 0.001


def spmdiff_efficient(adj1: SparseTensor, adj2: SparseTensor, keep_val: bool = False) -> SparseTensor:
    """
    Efficiently return the elements in adj1 but not in adj2.
    """
    # Ensure the input tensors have the same dimensions
    assert adj1.sizes() == adj2.sizes(), "Sparse tensors must have the same dimensions"
    
    row1, col1, val1 = adj1.coo()
    row2, col2, val2 = adj2.coo()

    keys1 = row1 * adj1.size(1) + col1  # Encode 2D indices into unique 1D keys
    keys2 = row2 * adj2.size(1) + col2

    mask1 = ~torch.isin(keys1, keys2)  # Mask elements in adj1 not found in adj2
    row_diff = row1[mask1]
    col_diff = col1[mask1]

    val1 = val1 if val1 is not None else torch.ones_like(row1, dtype=torch.float)
    val2 = val2 if val2 is not None else torch.ones_like(row2, dtype=torch.float)
            
    if keep_val:
        val_diff = val1[mask1]
        return SparseTensor.from_edge_index(
            torch.stack([row_diff, col_diff], dim=0),
            val_diff,
            adj1.sizes()
        )
    else:
        return SparseTensor.from_edge_index(
            torch.stack([row_diff, col_diff], dim=0),
            None,
            adj1.sizes()
        )


@torch.no_grad()
def valid_cn(encoder, predictor, data, splits, batch_size, device):
         
    encoder.eval()
    predictor.eval()

    pos_valid_edge = splits['pos_edge_label_index'].to(device)
    neg_valid_edge = splits['neg_edge_label_index'].to(device)
    pos_edge_label = splits['pos_edge_score'].to(device)
    neg_edge_label = splits['neg_edge_score'].to(device)

    for perm in tqdm(DataLoader(range(pos_valid_edge.size(1)), batch_size,
                           shuffle=True), desc='Valid'):
        
        edge = pos_valid_edge[:, perm]
        
        h = encoder(data.x, data.full_adj_t, data.edge_weight)
        neg_edge = neg_valid_edge[:, perm]
        
        pos_pred = predictor(h, edge).squeeze()
        neg_pred = predictor(h, neg_edge).squeeze()

        pos_loss = F.mse_loss(pos_pred, pos_edge_label[perm])
        neg_loss = F.mse_loss(neg_pred, neg_edge_label[perm])
        loss = pos_loss + neg_loss

    return loss.item()


def train_cn(encoder, predictor, optimizer, data, split_edge, batch_size, mask_target, step):
    encoder.train()
    predictor.train()
    
    device = data.x.device

    pos_train_edge = split_edge['train']['pos_edge_label_index'].to(device)
    neg_edge_epoch = split_edge['train']['neg_edge_label_index'].to(device)
    pos_edge_label = split_edge['train']['pos_edge_score'].to(device)
    neg_edge_label = split_edge['train']['neg_edge_score'].to(device)

    optimizer.zero_grad()
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    for perm in tqdm(DataLoader(range(pos_train_edge.size(1)), batch_size,
                           shuffle=True), desc='Train'):
        
        edge = pos_train_edge[:, perm]
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = adj_t.to(device)
            adj_t = spmdiff_efficient(adj_t, target_adj, keep_val=True)
        else:
            adj_t = data.adj_t
        
        adj_t = adj_t.to(device)
        # h = encoder(data.x, adj_t)
        h = encoder(data.x, data.adj_t, data.edge_weight)
        neg_edge = neg_edge_epoch[:, perm]
        
        pos_pred = predictor(h, edge).squeeze()
        neg_pred = predictor(h, neg_edge).squeeze()

        pos_loss = F.mse_loss(pos_pred, pos_edge_label[perm])
        neg_loss = F.mse_loss(neg_pred, neg_edge_label[perm])
        loss = pos_loss + neg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({f"Train/loss": loss.item()}, step=step)
        wandb.log({f"Train/pos_loss": pos_loss.item()}, step=step)
        wandb.log({f"Train/neg_loss": pos_loss.item()}, step=step)
        step += 1
        
    return loss.item()



def experiment_loop(args: argparse.Namespace, 
                    splits: Dict[str, Dict[str, Any]],
                    data: DataLoader):

    # Initialize model configuration
    with open(ROOT + '/exp1_gnn.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)

    cfg_encoder = eval(f'cfg.model.{args.model}')
    cfg_decoder = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
    
    pprint(cfg_encoder)
    pprint(cfg_decoder)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    cfg_encoder.in_channels = data.x.size(-1)
    cfg_encoder.num_nodes = data.num_nodes
        
    if args.model == 'LINKX':
        encoder = LINKX(
            num_nodes=cfg_encoder.num_nodes,
            in_channels=cfg_encoder.in_channels,
            hidden_channels=cfg_encoder.hidden_channels,
            out_channels=cfg_encoder.out_channels,
            num_layers=cfg_encoder.num_layers,
            num_edge_layers=cfg_encoder.num_edge_layers,
            num_node_layers=cfg_encoder.num_node_layers,
        )

    if args.model == 'Custom_GAT':
        encoder = Custom_GAT(cfg_encoder.in_channels,
                              cfg_encoder.hidden_channels,
                              cfg_encoder.out_channels,
                              cfg_encoder.num_layers,
                              cfg_encoder.dropout,
                              cfg_encoder.heads,
                              )
        
    if args.model == 'Custom_GCN':
        encoder = Custom_GCN(cfg_encoder.in_channels,
                              cfg_encoder.hidden_channels,
                              cfg_encoder.out_channels,
                              cfg_encoder.num_layers,
                              cfg_encoder.dropout,
                              )
        
    if args.model == 'GraphSAGE':
        encoder = GraphSAGE(cfg_encoder.in_channels,
                               cfg_encoder.hidden_channels,
                               cfg_encoder.out_channels,
                               cfg_encoder.num_layers,
                               cfg_encoder.dropout,
                               )
        
    if args.model == 'Custom_GIN':
        encoder = Custom_GIN(cfg_encoder.in_channels,
                              cfg_encoder.hidden_channels,
                              cfg_encoder.out_channels,
                              cfg_encoder.num_layers,
                              cfg_encoder.dropout,
                              cfg_encoder.mlp_layer
                              )
        
    if args.model == 'HLGNN':
        encoder = HLGNN(cfg_encoder.in_channels,
                        hidden_channels=256,
                        out_channels=256,
                        K=3,
                        dropout=0.3,
                        alpha=0.5, 
                        init='KI'
                        )
        
        cfg_decoder.out_channels = 256
        cfg_decoder.score_hidden_channels = 256
        
    decoder = LinkPredictor(cfg_encoder.out_channels,
                cfg_decoder.score_hidden_channels,
                cfg_decoder.score_out_channels,
                cfg_decoder.score_num_layers,
                cfg_decoder.score_dropout)

    encoder.reset_parameters()
    decoder.reset_parameters()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')


    step = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        data.adj_t = SparseTensor.from_edge_index(splits['train'].edge_index)
        train_loss = train_cn(
            encoder, decoder, optimizer, data, splits, args.batch_size, True, step
        ) 
        
        if epoch % 1 == 0:
            valid_loss = valid_cn(encoder, decoder, data, splits['valid'], args.batch_size, device)
            
            print(
                f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, '
                f'valid loss: {valid_loss:.4f}, '
                f'Cost Time: {time.time() - start:.4f}s')
            
            if args.use_early_stopping:
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Training stopped early!")
                    break
                
    data.adj_t = data.full_adj_t
    test_loss = valid_cn(encoder, decoder, data, splits['test'], args.batch_size, device)
    # Save results to CSV with mean and variance
    save_to_csv(f'./results/{args.dataset}/{args.model}2{args.h_key}_{args.dataset}.csv',
                args.model,
                args.nodefeat,
                args.h_key,
                test_loss)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for training.")

    # Training parameters 
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--dataset', type=str, default="Cora", help="Dataset to use.")
    parser.add_argument('--batch_size', type=int, default=2**12, help="Batch size for training.")
    parser.add_argument('--h_key', type=str, default="CN", help="Heuristic key to use.")
    parser.add_argument('--model', type=str, default="HLGNN", 
                        choices = ["LINKX", "Custom_GAT", "Custom_GCN", "GraphSAGE", "Custom_GIN"], 
                        help="Model type to use.")
    parser.add_argument('--nodefeat', type=str, default="adjacency", 
                        choices = ["one-hot", "random", "quasi-orthogonal", "adjacency", "original"],
                        help="Node feature type to use.")
    # Early stopping
    parser.add_argument('--use_early_stopping', action='store_true', help="Enable early stopping.")

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for optimizer.")
    parser.add_argument('--generate_dataset', type=bool, default=True, help="enable generate dand save dataset.")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, splits = loaddataset(args.dataset, None)
    data = data.to(device) 

    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )
        
    heuristic = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA,
        "PPR": Ben_PPR,
        "katz": katz_apro,
        "shortest_path": shortest_path
    }

    if args.generate_dataset:
        for key in splits:
            pos_edge_score, _ = heuristic[args.h_key](
                A, splits[key]['pos_edge_label_index'], batch_size=args.batch_size
            )
            neg_edge_score, _ = heuristic[args.h_key](
                A, splits[key]['neg_edge_label_index'], batch_size=args.batch_size
            )
            
            # Normalize the scores
            max_score = pos_edge_score.max()
            splits[key]['pos_edge_score'] = pos_edge_score / max_score
            splits[key]['neg_edge_score'] = neg_edge_score / max_score

        with open(f"{args.dataset}_{args.h_key}_data.pkl", "wb") as f:
            pickle.dump(splits, f)
    else:
        # Replace 'args.dataset' and 'args.h_key' with your specific file naming variables or values.
        file_path = ROOT + f"/{args.dataset}_{args.h_key}_data.pkl"

        # Open the file in 'read binary' mode and load the data
        with open(file_path, "rb") as f:
            splits = pickle.load(f)
        print(f'dataset is loaded: {splits.keys()}')
    
    
    for i in range(NUM_SEED):
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(i)
        random.seed(i)


        if args.nodefeat == 'one-hot':
            data.x = torch.eye(data.num_nodes, data.num_nodes).to(device)
        elif args.nodefeat == 'random':
            dim_feat = data.x.size(1)
            data.x = torch.randn(data.num_nodes, data.num_nodes).to(device)
        elif args.nodefeat == 'quasi-orthogonal':
            pass 
        elif args.nodefeat == 'adjacency':
            A_dense = A.toarray()
            A_tensor = torch.tensor(A_dense)
            data.x = A_tensor.float().to(device)
        elif args.nodefeat == 'original':
            pass
        else:
            raise NotImplementedError(
                f'nodefeat: {args.nodefeat} is not implemented.'
            )
        print(f"initialize project: {args.model}_{args.h_key}_{args.dataset}_{args.nodefeat}_{i}")

        wandb.init(project=f"graph-link-prediction-{args.dataset}-{args.h_key}",     
                    name = f"run_{args.model}_{args.h_key}_{args.dataset}_{args.nodefeat}_{i}",
                    config={
                    "epochs": args.epochs,
                    "dataset": args.dataset,
                    "batch_size": args.batch_size,
                    "h_key": args.h_key,
                    "model": args.model,
                    "lr": args.lr,
                    "weight_decay": 0
        })
        
        config = wandb.config
        experiment_loop(args, splits, data)
        wandb.finish()
        # TODO training is instable for LINKX and Custom_GCN try lr log3
        # TODO dataloader into repository 
        # TODO review synehtic graph dataloader 
