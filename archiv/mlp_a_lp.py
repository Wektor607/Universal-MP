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
from baselines.heuristic import CN, AA, RA
from baselines.GNN import GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, GAE4LP, InnerProduct, mlp_score
from archiv.mlp_heuristic_main import EarlyStopping
from utils import  EarlyStopping, visualize
from baselines.heuristic import CN as CommonNeighbor

class NNet(nn.Module):

    def __init__(self, n_in, n_out, hlayers=(128, 256, 128)):
        super(NNet, self).__init__()
        self.n_hlayers = len(hlayers)
        self.fcs = nn.ModuleList([nn.Linear(n_in, hlayers[i]) if i == 0 else
                                  nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
                                  nn.Linear(hlayers[i-1], hlayers[i]) for i in range(self.n_hlayers+1)])

    def forward(self, x):
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = torch.sigmoid(self.fcs[-1](x))
        return x


class MLPPolynomialFeatures(nn.Module):
    def __init__(self, num_nodes, 
                 dim_feat, 
                 hidden_dim, 
                 K, 
                 A, 
                 dropout, 
                 use_nodefeat):
        super(MLPPolynomialFeatures, self).__init__()
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
                                            nn.Linear(hlayers[i-1], n_out) if i == self.n_hlayers else
                                            nn.Linear(hlayers[i-1], hlayers[i]) for i in range(self.n_hlayers+1)])

        self.dropout = nn.Dropout(p=self.dropout)
        A = torch.tensor(A.toarray(), dtype=torch.float32)

        # high space complexity in large graph
        self.A_powers = [A]  # A^1
        for _ in range(1, K): # K, order of A 
            self.A_powers.append(torch.matmul(self.A_powers[-1], A))  # A^n
        
        # element-wise mult of A
        self.device = next(self.parameters()).device        

    def forward(self, i, j, x):
        assert torch.all((i >= 0) & (i < self.A_powers[0].shape[0])), "Index i is out of bounds."
        assert torch.all((j >= 0) & (j < self.A_powers[0].shape[0])), "Index j is out of bounds."
        
        self.A_powers = [A_n.to(self.device) for A_n in self.A_powers]

        # dot products for each power of A
        A_emb = [A_n[i] * A_n[j] for A_n in self.A_powers]
        A_embK = torch.cat(A_emb, dim=1)  

        if self.use_nodefeat:
            text_emb = x[i]* x[j]
            x = torch.cat((A_embK, text_emb), dim=1)
        else:
            x = A_embK
            
        for i in range(self.n_hlayers):
            x = F.relu(self.mlp_module[i](x))
            x = self.dropout(x)
        x = torch.sigmoid(self.mlp_module[-1](x))

        return x.view(-1)


def train(model, optimizer, data, splits, device, A, batch_size=512):
    model.train()
    optimizer.zero_grad()
    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_label'].to(device)
    neg_edge_label = splits['train']['neg_edge_label'].to(device)

    total_loss = 0
    pos_preds, neg_preds = [], []

    for i in range(0, pos_edge_index.size(1), batch_size):
        batch_pos_edge_index = pos_edge_index[:, i:i+batch_size]
        batch_pos_pred = model(batch_pos_edge_index[0], batch_pos_edge_index[1],
                               data.x)
        pos_preds.append(batch_pos_pred)
        batch_pos_loss = -torch.log(batch_pos_pred + 1e-15).mean()
        total_loss += batch_pos_loss

    for i in range(0, neg_edge_index.size(1), batch_size):
        batch_neg_edge_index = neg_edge_index[:, i:i+batch_size]
        batch_neg_pred = model(batch_neg_edge_index[0], batch_neg_edge_index[1],
                               data.x)
        neg_preds.append(batch_neg_pred)
        batch_neg_loss = -torch.log(1 - batch_neg_pred + 1e-15).mean()
        total_loss += batch_neg_loss

    total_loss.backward()
    optimizer.step()

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())
    return auc


def valid(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)
    pos_label = splits['valid']['pos_edge_label'].to(device)
    neg_label = splits['valid']['neg_edge_label'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x)
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x)

    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    # Compute AUC
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())

    return auc


@torch.no_grad()
def test(model, data, splits, device, A):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_label = splits['test']['pos_edge_label'].to(device)
    neg_label = splits['test']['neg_edge_label'].to(device)

    # Forward pass
    pos_pred = model(pos_edge_index[0], pos_edge_index[1], data.x)
    neg_pred = model(neg_edge_index[0], neg_edge_index[1], data.x)

    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    # Compute AUC
    auc = roc_auc_score(all_labels.cpu().detach().numpy(), all_preds.cpu().detach().numpy())
    visualize(pos_pred, pos_label, 'pos.png')
    visualize(neg_pred, neg_label, 'neg.png')

    return auc


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="Citeseer")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--use_nodefeat', type=bool, default=True)
    parser.add_argument('--node_feature', type=str, default='one-hot')
    
    args = parser.parse_args()
    return args

def save_to_csv(file_path, 
                model_name, 
                use_nodefeat, 
                nodefeat,
                test_auc):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'Feature', 'Nodefeat', 'Test_AUC'])
        writer.writerow([model_name+'_with_features', nodefeat, use_nodefeat, test_auc])
        

def experiment_loop():
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
        else:
            raise NotImplementedError(f'node_feature: {args.node_feature} is not implemented.')

    model = MLPPolynomialFeatures(data.num_nodes, 
                                  data.x.size(1), 
                                  hidden_dim=64, 
                                  K=2, 
                                  A=A, 
                                  dropout=0.1,
                                  use_nodefeat=args.use_nodefeat).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        auc = train(model, optimizer, data, splits, device, A, batch_size=args.batch_size)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}')
        val_auc  = valid(model, data, splits, device, A)
        print(f'Validation AUC: {val_auc:.4f}')
        test_auc = test(model, data, splits, device, A)
        print(f'Test AUC: {test_auc:.4f}')
        if args.early_stopping:
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
        print(f'Time taken: {time.time() - start:.2f}s')
        
    test_auc = test(model, data, splits, device, A)
    print(f'Test Result: AUC: {test_auc:.4f}')
    save_to_csv(f'./results/lp_results_{args.dataset}.csv', args.model, args.use_nodefeat, args.node_feature, test_auc)
    print(f'Saved results.')


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
        else:
            raise NotImplementedError(f'node_feature: {args.node_feature} is not implemented.')

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

    model = MLPPolynomialFeatures(data.num_nodes, 
                                  data.x.size(1), 
                                  hidden_dim=64, 
                                  K=2, 
                                  A=A, 
                                  dropout=0.1,
                                  use_nodefeat=args.use_nodefeat).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        auc = train(model, optimizer, data, splits, device, A, batch_size=args.batch_size)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}')
        val_auc  = valid(model, data, splits, device, A)
        print(f'Validation AUC: {val_auc:.4f}')
        test_auc = test(model, data, splits, device, A)
        print(f'Test AUC: {test_auc:.4f}')
        if args.early_stopping:
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
        print(f'Time taken: {time.time() - start:.2f}s')
        
    test_auc = test(model, data, splits, device, A)
    print(f'Test Result: AUC: {test_auc:.4f}')
    save_to_csv(f'./results/lp_results_{args.dataset}.csv', args.model, args.use_nodefeat, args.node_feature, test_auc)
    print(f'Saved results.')


if __name__ == "__main__":
    args = parseargs()
    args.use_nodefeat = True
    for args.node_feature in ['original', 'one-hot', 'random', 'adjacency']:
        for i in range(5):
            experiment_loop()