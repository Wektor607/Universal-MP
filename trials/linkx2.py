
import os
import sys
import csv
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import numpy as np
from yacs.config import CfgNode
from pprint import pprint 
from train_utils import train, valid, test

from baselines.utils import loaddataset

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from baselines.heuristic import AA, RA, Ben_PPR, katz_apro, shortest_path
from baselines.heuristic import CN as CommonNeighbor
from baselines.GNN import  LinkPredictor, GAE4LP
from yacs.config import CfgNode as CN
from archiv.mlp_heuristic_main import EarlyStopping
from baselines.LINKX import LINKX
from train_utils import train, valid, test
from utils import save_to_csv, visualize

class Config:
    def __init__(self):
        self.epochs = 100
        self.dataset = "ddi"
        self.batch_size = 8192
        self.h_key = "CN"
        self.gnn = "gcn"
        self.model = "LINKX"
        self.use_feature = False
        self.node_feature = 'original'  # 'one-hot', 'random', 
                                        # 'quasi-orthogonal', 'adjacency'
        self.use_early_stopping = True


def init_LINKX(
    cfg_encoder: CN,
    cfg_decoder: CN,
    m_name: str):
    # split them into two models 
    # https://github.com/Barcavin/efficient-node-labelling/blob/master/main.py
    
    if m_name in ['LINKX']:
        encoder = LINKX(
            num_nodes=cfg_encoder.num_nodes,
            in_channels=cfg_encoder.in_channels,
            hidden_channels=cfg_encoder.hidden_channels,
            out_channels=cfg_encoder.out_channels,
            num_layers=cfg_encoder.num_layers,
            num_edge_layers=cfg_encoder.num_edge_layers,
            num_node_layers=cfg_encoder.num_node_layers,
        )

    decoder = LinkPredictor(cfg_encoder.out_channels,
                        cfg_decoder.score_hidden_channels,
                        cfg_decoder.score_out_channels,
                        cfg_decoder.score_num_layers,
                        cfg_decoder.score_dropout)

    return GAE4LP(encoder=encoder, decoder=decoder)



def experiment_loop(args: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and set up data
    data, splits = loaddataset(args.dataset, None)
    data = data.to(device)

    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )

    # Config node feature
    if args.node_feature == 'one-hot':
        data.x = torch.eye(data.num_nodes, data.num_nodes).to(device)
    elif args.node_feature == 'random':
        dim_feat = data.x.size(1)
        data.x = torch.randn(data.num_nodes, dim_feat).to(device)
    elif args.node_feature == 'adjacency':
        A_dense = A.toarray()
        A_tensor = torch.tensor(A_dense)
        data.x = A_tensor.float().to(device)
    elif args.node_feature == 'original':
        pass
    else:
        raise NotImplementedError(
            f'node_feature: {args.node_feature} is not implemented.'
        )

    # Initialize model configuration
    with open('yamls/cora/heart_gnn_models.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)

    cfg_encoder = eval(f'cfg.model.{args.model}')
    cfg_decoder = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
    
    pprint(cfg_encoder)
    pprint(cfg_decoder)

    if not hasattr(splits['train'], 'x') or splits['train'].x is None:
        cfg_encoder.in_channels = 1024
    else:
        cfg_encoder.in_channels = data.x.size(1)

    heuristic = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA,
        "PPR": Ben_PPR,
        "katz": katz_apro,
        "shortest_path": shortest_path
    }

    # Heuristic scores
    for key in splits:
        pos_edge_score, _ = heuristic[args.h_key](
            A, splits[key]['pos_edge_label_index'], batch_size=args.batch_size
        )
        neg_edge_score, _ = heuristic[args.h_key](
            A, splits[key]['neg_edge_label_index'], batch_size=args.batch_size
        )
        splits[key]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[key]['neg_edge_score'] = torch.sigmoid(neg_edge_score)

    # refine it as class and save into .npz
    # Store results of each experiment
    early_stopping = EarlyStopping(patience=5, verbose=True)

    if args.model == 'LINKX':
        cfg_encoder.in_channels = data.x.size(-1)
        cfg_encoder.num_nodes = data.num_nodes

        encoder = LINKX(
            num_nodes=cfg_encoder.num_nodes,
            in_channels=cfg_encoder.in_channels,
            hidden_channels=cfg_encoder.hidden_channels,
            out_channels=cfg_encoder.out_channels,
            num_layers=cfg_encoder.num_layers,
            num_edge_layers=cfg_encoder.num_edge_layers,
            num_node_layers=cfg_encoder.num_node_layers,
        )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(
            encoder, decoder, optimizer, data, splits, device, args.batch_size
        )

        # Validate every 20 epochs
        if epoch % 20 == 0:
            valid_loss = valid(encoder, decoder, data, splits, device, args.batch_size)
            print(
                f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, '
                f'valid loss: {valid_loss:.4f}, '
                f'Cost Time: {time.time() - start:.4f}s')

            if args.use_early_stopping:
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Training stopped early!")
                    break

    test_loss = test(encoder, decoder, data, splits, device, args.batch_size)

    # Save results to CSV with mean and variance
    save_to_csv(f'./results/LINKX2{args.h_key}_{args.dataset}.csv',
                args.model,
                args.node_feature,
                args.h_key,
                test_loss)

    print('Saved mean and variance of test loss to CSV.')


if __name__ == "__main__":

    args = Config()
    for model in ['LINKX']:
        args.model = model
        for node_feature in ['original', 'one-hot', 'random', 'adjacency']:
            args.node_feature = node_feature
            for i in range(3):
                experiment_loop(args)