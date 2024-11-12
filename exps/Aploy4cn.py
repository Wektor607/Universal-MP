import argparse
import csv
import os
import sys
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
from yacs.config import CfgNode

from sklearn.metrics import roc_auc_score
from baselines.MLP import MLPPolynomialFeatures
from baselines.utils import loaddataset
from baselines.heuristic import CN, AA, RA, Ben_PPR, katz_apro
from baselines.GNN import GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, GAE_forall, InnerProduct, mlp_score
from archiv.mlp_heuristic_main import EarlyStopping
from utils import EarlyStopping, visualize
from baselines.heuristic import CN as CommonNeighbor


class Config:
    def __init__(self):
        self.epochs = 2000
        self.dataset = "ddi"
        self.batch_size = 8192
        self.model = "MLP"
        self.early_stopping = True
        self.use_feature = True
        self.node_feature = 'one-hot'
        self.heuristic = "PPR"
        self.K = 2


class NNet(nn.Module):

    def __init__(self, n_in, n_out, hlayers=(128, 256, 128)):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                  nn.Linear(hlayers[i - 1], n_out) if i == self.n_hlayers else
                                  nn.Linear(hlayers[i - 1], hlayers[i]) for i in range(self.n_hlayers + 1)])

    def forward(self, x):
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = torch.sigmoid(self.fcs[-1](x))
        return x


class APoly_MLP(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 dim_feat,
                 hidden_dim,
                 K,
                 A,
                 dropout,
                 use_nodefeat):
        super(APoly_MLP, self).__init__()
        """data.num_nodes, data.x.size(1), hidden_channels=64, K=2, A=A).to(device)"""
        # self.num_series = K
        self.dropout = dropout

        # params for MLPs
        self.n_hlayers = 3
        self.use_nodefeat = use_nodefeat
        if self.use_nodefeat:
            n_in = num_nodes * K + dim_feat  # size of concated A^k and features
        else:
            n_in = num_nodes * K

        n_out = 1
        hlayers = (hidden_dim, hidden_dim, hidden_dim)
        self.mlp_module = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                         nn.Linear(hlayers[i - 1], n_out) if i == self.n_hlayers else
                                         nn.Linear(hlayers[i - 1], hlayers[i]) for i in range(self.n_hlayers + 1)])
        self.dropout = nn.Dropout(p=self.dropout)

        self.device = next(self.parameters()).device

        A = torch.tensor(A.toarray(), dtype=torch.float32).to(self.device)

        # TODO high space complexity in large graph
        self.A_powers = [A]  # A^1
        for _ in range(1, K):  # K, order of A
            self.A_powers.append(torch.matmul(self.A_powers[-1], A))  # A^n
        for i in range(len(self.mlp_module)):
            self.mlp_module[i] = self.mlp_module[i].to(self.device)

    def forward(self, src, tar, x):
        assert torch.all((src >= 0) & (src < self.A_powers[0].shape[0])), "Index i is out of bounds."
        assert torch.all((tar >= 0) & (tar < self.A_powers[0].shape[0])), "Index j is out of bounds."

        self.A_powers = [A_n.to(self.device) for A_n in self.A_powers]

        # dot products for each power of A
        src, tar = src.to(self.device), tar.to(self.device)
        A_emb = [A_n[src] * A_n[tar] for A_n in self.A_powers]

        A_embK = torch.cat(A_emb, dim=1)

        if self.use_nodefeat:
            text_emb = x[src] * x[tar]
            x = torch.cat((A_embK.to(self.device), text_emb.to(self.device)), dim=1)
        else:
            x = A_embK

        for i in range(self.n_hlayers):
            x = x.to(self.mlp_module[i].weight.device)
            x = F.relu(self.mlp_module[i](x))
            x = self.dropout(x)
        x = torch.sigmoid(self.mlp_module[-1](x))
        return x.view(-1)


def train(model, optimizer, data, splits, device, batch_size=2048):
    model.train()
    total_loss = 0

    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_score'].to(device)
    neg_edge_label = splits['train']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx].to(device)
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx].to(device)
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx].to(device)
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx].to(device)

        optimizer.zero_grad()

        pos_pred = model(batch_pos_edge_index[0], batch_pos_edge_index[1], data.x)
        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)

        neg_pred = model(batch_neg_edge_index[0], batch_neg_edge_index[1], data.x)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)

        batch_loss = pos_loss + neg_loss
        batch_loss.backward()

        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / num_batches


def valid(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)
    pos_label = splits['valid']['pos_edge_score'].to(device)
    neg_label = splits['valid']['neg_edge_score'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x)
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x)

    pos_pred = model(pos_edge_index[0], pos_edge_index[1],
                     data.x)
    pos_loss = F.mse_loss(pos_pred, pos_label)
    neg_pred = model(neg_edge_index[0], neg_edge_index[1],
                     data.x)
    neg_loss = F.mse_loss(neg_pred, neg_label)
    total_loss = pos_loss + neg_loss

    return total_loss.item()


@torch.no_grad()
def test(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_label = splits['test']['pos_edge_score'].to(device)
    neg_label = splits['test']['neg_edge_score'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x)
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x)

    pos_pred = model(pos_edge_index[0], pos_edge_index[1],
                     data.x)
    pos_loss = F.mse_loss(pos_pred, pos_label)

    neg_pred = model(neg_edge_index[0], neg_edge_index[1],
                     data.x)
    neg_loss = F.mse_loss(neg_pred, neg_label)
    total_loss = pos_loss + neg_loss

    return total_loss.item()


def save_to_csv(file_path,
                K,
                use_nodefeat,
                nodefeat,
                test_metric):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Feature', 'Nodefeat', 'Test_MSE'])
        writer.writerow([f'APoly4{args.heuristic}_{K}', nodefeat, args.heuristic, test_metric])


def experiment_loop_cn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, splits = loaddataset(args.dataset, True)
    data = data.to(device)
    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )

    if args.use_nodefeat:
        if args.node_feature == 'one-hot':
            data.x = torch.eye(data.num_nodes, data.num_nodes).to(device)
        elif args.node_feature == 'random':
            dim_feat = data.x.size(1)
            data.x = torch.randn(data.num_nodes, dim_feat).to(device)
        elif args.node_feature == 'quasi-orthogonal':
            pass
        elif args.node_feature == 'adjacency':
            A_dense = A.toarray()
            A_tensor = torch.tensor(A_dense)
            data.x = A_tensor.float().to(device)
        elif args.node_feature == 'original':
            pass
        elif args.node_feature == 'none':
            pass
        else:
            raise NotImplementedError(f'{args.node_feature} is missing.')

    method_dict = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA,
        "PPR": Ben_PPR,
        "katz": katz_apro,
    }
    for split in splits:
        pos_edge_score, _ = method_dict[args.heuristic](
            A, splits[split]['pos_edge_label_index'],
            batch_size=args.batch_size
        )
        neg_edge_score, _ = method_dict[args.heuristic](
            A, splits[split]['neg_edge_label_index'],
            batch_size=args.batch_size
        )
        splits[split]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[split]['neg_edge_score'] = torch.sigmoid(neg_edge_score)

    model = APoly_MLP(data.num_nodes,
                      data.x.size(1),
                      hidden_dim=64,
                      K=args.K,
                      A=A,
                      dropout=0.1,
                      use_nodefeat=args.use_nodefeat).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_mse = train(model, optimizer, data, splits, device)
        print(f'Epoch: {epoch:03d}, MSE: {train_mse:.4f}')
        val_mse = valid(model, data, splits, device, A)
        print(f'Validation MSE: {val_mse:.4f}')
        if args.early_stopping:
            early_stopping(val_mse)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
        print(f'Time taken: {time.time() - start:.2f}s')

    test_mse = test(model, data, splits, device, A)
    print(f'Test Result: MSE: {test_mse:.4f}')
    save_to_csv(f'./results/APoly4{args.heuristic}_{args.dataset}.csv',
                args.K,  # K
                args.use_nodefeat,
                args.node_feature,
                test_mse)
    print(f'Saved to ./results/APoly4{args.heuristic}_{args.dataset}.csv')


if __name__ == "__main__":
    args = Config()
    for args.K in [2, 3]:
        args.use_nodefeat = True
        for args.node_feature in ['one-hot', 'random', 'adjacency', 'original']:
            for i in range(5):
                experiment_loop_cn()
        args.use_feature = False
        for i in range(5):
            args.node_feature = 'none'
            experiment_loop_cn()