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
from sklearn.metrics import roc_auc_score
from baselines.MLP import MLPPolynomialFeatures
from baselines.utils import loaddataset
from baselines.heuristic import AA, RA, Ben_PPR, katz_apro
from baselines.heuristic import CN as CommonNeighbor
from baselines.GNN import GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, GAE_forall, InnerProduct, mlp_score
from yacs.config import CfgNode as CN
from archiv.mlp_heuristic_main import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

class Config:
    def __init__(self):
        self.epochs = 100
        self.dataset = "ddi"
        self.batch_size = 512
        self.heuristic = "CN"
        self.gnn = "gcn"
        self.model = "GIN_Variant"
        self.use_feature = False
        self.node_feature = 'adjacency' # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True



def create_GAE_model(cfg_model: CN,
                     cfg_score: CN,
                     model_name: str):
    if model_name in {'GAT', 'VGAE', 'GAE', 'GraphSage'}:
        raise NotImplementedError('Current model does not exist')
        # model = create_model(cfg_model)

    elif model_name == 'GAT_Variant':
        encoder = GAT_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.heads,
                              )
    elif model_name == 'GCN_Variant':
        encoder = GCN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              )
    elif model_name == 'SAGE_Variant':
        encoder = SAGE_Variant(cfg_model.in_channels,
                               cfg_model.hidden_channels,
                               cfg_model.out_channels,
                               cfg_model.num_layers,
                               cfg_model.dropout,
                               )
    elif model_name == 'GIN_Variant':
        encoder = GIN_Variant(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.mlp_layer
                              )
    if cfg_score.product == 'dot':
        decoder = mlp_score(cfg_model.out_channels,
                            cfg_score.score_hidden_channels,
                            cfg_score.score_out_channels,
                            cfg_score.score_num_layers,
                            cfg_score.score_dropout,
                            cfg_score.product)
    elif cfg_score.product == 'inner':
        decoder = InnerProduct()

    else:
        # Without this else I got: UnboundLocalError: local variable 'model' referenced before assignment
        raise ValueError('Current model does not exist')

    return GAE_forall(encoder=encoder, decoder=decoder)


def train(model, optimizer, data, splits, device, epoch):
    model.train()
    optimizer.zero_grad()
    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_score'].to(device)
    neg_edge_label = splits['train']['neg_edge_score'].to(device)

    # Forward pass
    z = model.encode(data.x, data.edge_index)

    # pred for pos, neg edges
    pos_pred = model.decode(z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_pred = model.decode(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # regression loss
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss
    loss.backward()

    optimizer.step()
    # visualize(pos_pred, pos_edge_label, save_path='./visualization_pos_train.png')
    # visualize(neg_pred, neg_edge_label, save_path='./visualization_neg_train.png')
    return loss.item()


@torch.no_grad()
def valid(model, data, splits, device, epoch):
    model.eval()

    # Positive and negative edges for validation
    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['valid']['pos_edge_score'].to(device)
    neg_edge_label = splits['valid']['neg_edge_score'].to(device)

    # Forward pass
    z = model.encode(data.x, data.edge_index)

    # Predict scores for both positive and negative edges
    pos_pred = model.decode(z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_pred = model.decode(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # Compute regression loss (MSE)
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss
    all_preds = torch.cat([pos_pred, neg_pred], dim=0)
    all_labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

    return loss.item()


@torch.no_grad()
def test(model, data, splits, device):
    model.eval()

    # Positive and negative edges for test
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)

    # Labels for positive and negative edges (continuous regression labels)
    pos_edge_label = splits['test']['pos_edge_score'].to(device)
    neg_edge_label = splits['test']['neg_edge_score'].to(device)

    # Forward pass
    z = model.encode(data.x, data.edge_index)

    # Predict scores for both positive and negative edges
    pos_pred = model.decode(z[pos_edge_index[0]], z[pos_edge_index[1]])
    neg_pred = model.decode(z[neg_edge_index[0]], z[neg_edge_index[1]])
    visualize(pos_pred, pos_edge_label, save_path = './visualization_pos.png')
    visualize(neg_pred, neg_edge_label, save_path = './visualization_neg.png')

    # Compute regression loss (MSE)
    pos_loss = F.mse_loss(pos_pred, pos_edge_label)
    neg_loss = F.mse_loss(neg_pred, neg_edge_label)
    loss = pos_loss + neg_loss
    return loss.item()


def save_to_csv(file_path: object,
                model_name: object,
                node_feat: object,
                test_loss: object,
                heuristic: object) -> object:
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'NodeFeat', 'Heuristic', 'Test_Loss'])
        writer.writerow([model_name, node_feat, heuristic, test_loss])
    print(f'Saved {model_name, node_feat, heuristic, test_loss} to {file_path}')
    
     
def visualize(pred, true_label, save_path = './visualization.png'):

    pred = pred.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(true_label)), true_label, color='#A6CEE3', label='True Score', alpha=0.6)
    plt.scatter(np.arange(len(pred)), pred, color='#B2DF8A', label='Prediction', alpha=0.6)

    plt.title('Predictions vs True Score')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    print(f"Visualization saved at {save_path}")
    return 




def experiment_loop(args: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data, splits = loaddataset(args.dataset, True)
    data = data.to(device)

    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )
        
    # change node feature
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

    # initialize model 
    with open('./yamls/cora/heart_gnn_models.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)
        
    cfg_model = eval(f'cfg.model.{args.model}')
    cfg_score = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model
    
    if not hasattr(splits['train'], 'x') or splits['train'].x is None:
        cfg_model.in_channels = 1024
    else:
        cfg_model.in_channels = data.x.size(1)

    method_dict = {
        "CN": CommonNeighbor,
        "AA": AA,
        "RA": RA,
        "PPR": Ben_PPR,
        "katz": katz_apro,
    }
    for split in splits:
        pos_edge_score, _ = method_dict[args.heuristic](A, splits[split]['pos_edge_label_index'],
                                                        batch_size=args.batch_size)
        neg_edge_score, _ = method_dict[args.heuristic](A, splits[split]['neg_edge_label_index'],
                                                        batch_size=args.batch_size)
        splits[split]['pos_edge_score'] = torch.sigmoid(pos_edge_score)
        splits[split]['neg_edge_score'] = torch.sigmoid(neg_edge_score)

    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    if args.model in ['GCN_Variant', 'GAT_Variant', 'SAGE_Variant', 'GIN_Variant']:
        model = create_GAE_model(cfg_model, cfg_score, args.model).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, data, splits, device, args.batch_size)
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f},  Cost Time: {time.time() - start:.4f}s')

        valid_loss= valid(model, data, splits, device, epoch)
        print(f'Train Loss: {valid_loss:.4f}')
        if args.use_early_stopping:
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Training stopped early!")
                break
            
    test_auc = test(model, data, splits, device)

    save_to_csv(f'./results/gcn4{args.heuristic}_{args.dataset}.csv', args.model, args.node_feature, test_auc,
                args.heuristic)
    


if __name__ == "__main__":
    
    args = Config()
    for model in ['GIN_Variant', 'SAGE_Variant', 'GCN_Variant']:
        args.model = model
        for node_feature in ['random', 'original', 'adjacency', 'one-hot']:
            args.node_feature = node_feature
            
            for i in range(5):
                experiment_loop(args)