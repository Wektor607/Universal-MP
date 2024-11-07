import argparse
import csv
import os, sys

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
from yacs.config import CfgNode

from graphgps.utils.ogbdataset import loaddataset
from graphgps.utils.heuristic import AA, RA
from graphgps.utils.heuristic import CN as CommonNeighbor
from graphgps.models.GNN import GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, GAE_forall, InnerProduct, mlp_score
from exps.cn_gcn_adjnode import create_GAE_model, train, valid, test, save_to_csv, visualize



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--heuristic', type=str, default="CN")
    parser.add_argument('--gnn', type=str, default="gcn")
    parser.add_argument('--model', type=str, default="GIN_Variant")
    parser.add_argument('--use_feature', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, splits = loaddataset(args.dataset, True)
    data = data.to(device)
    with open('./yamls/cora/heart_gnn_models.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)
    cfg_model = eval(f'cfg.model.{args.model}')
    
    if not hasattr(splits['train'], 'x') or splits['train'].x is None:
        cfg_model.in_channels = 1024
    else:
        cfg_model.in_channels = data.num_nodes
        
    cfg_score = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )
    method_dict = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA
    }
    for split in splits:
        pos_edge_score, _ = method_dict[args.heuristic](A, splits[split]['pos_edge_label_index'],
                                                        batch_size=args.batch_size)
        neg_edge_score, _ = method_dict[args.heuristic](A, splits[split]['neg_edge_label_index'],
                                                        batch_size=args.batch_size)
        splits[split]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[split]['neg_edge_score'] = torch.sigmoid(neg_edge_score)
        
    if not args.use_feature:
        data.x = torch.eye(data.num_nodes, data.num_nodes).to(device)

    model = create_GAE_model(cfg_model, cfg_score, args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(model, optimizer, data, splits, device, args.batch_size)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_loss = test(model, data, splits, device)
    save_to_csv(f'./results/test_results_{args.dataset}.csv', args.model, args.heuristic, test_loss)
    print(f'Saved results.')

