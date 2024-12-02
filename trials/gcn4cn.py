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
from baselines.GNN import Custom_GAT, Custom_GCN, GraphSAGE, GIN_Variant, GAE4LP, InnerProduct, LinkPredictor
from yacs.config import CfgNode as CN
from archiv.mlp_heuristic_main import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


class Config:
    def __init__(self):
        self.epochs = 100
        self.dataset = "ddi"
        self.batch_size = 8192
        self.heuristic = "PPR"
        self.gnn = "gcn"
        self.model = "GIN_Variant"
        self.use_feature = False
        self.node_feature = 'adjacency'  # 'one-hot', 'random', 'quasi-orthogonal'
        self.use_early_stopping = True


def create_GAE_model(cfg_model: CN,
                     cfg_score: CN,
                     model_name: str):
    if model_name in {'GAT', 'VGAE', 'GAE', 'GraphSage'}:
        raise NotImplementedError('Current model does not exist')
        # model = create_model(cfg_model)

    elif model_name == 'Custom_GAT':
        encoder = Custom_GAT(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              cfg_model.heads,
                              )
    elif model_name == 'Custom_GCN':
        encoder = Custom_GCN(cfg_model.in_channels,
                              cfg_model.hidden_channels,
                              cfg_model.out_channels,
                              cfg_model.num_layers,
                              cfg_model.dropout,
                              )
    elif model_name == 'GraphSAGE':
        encoder = GraphSAGE(cfg_model.in_channels,
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
        decoder = LinkPredictor(cfg_model.out_channels,
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

    return GAE4LP(encoder=encoder, decoder=decoder)


def train(model, optimizer, data, splits, device, epoch, batch_size=512):
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

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        optimizer.zero_grad()

        batch_edge_index = torch.cat([batch_pos_edge_index, batch_neg_edge_index], dim=1)

        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]])
        neg_pred = model.decode(z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]])

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        batch_loss.backward()

        total_loss += batch_loss.item()

        optimizer.step()

    return total_loss / num_batches


@torch.no_grad()
def valid(model, data, splits, device, epoch, batch_size=512):
    model.eval()

    total_loss = 0

    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['valid']['pos_edge_score'].to(device)
    neg_edge_label = splits['valid']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        batch_edge_index = torch.cat([batch_pos_edge_index, batch_neg_edge_index], dim=1)

        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]])
        neg_pred = model.decode(z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]])

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model, data, splits, device, batch_size=512):
    model.eval()

    total_loss = 0

    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['test']['pos_edge_score'].to(device)
    neg_edge_label = splits['test']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        batch_edge_index = torch.cat([batch_pos_edge_index, batch_neg_edge_index], dim=1)

        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]])
        neg_pred = model.decode(z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]])

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches


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


def visualize(pred, true_label, save_path='./visualization.png'):
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

    if args.model in ['Custom_GCN', 'Custom_GAT', 'GraphSAGE', 'GIN_Variant']:
        model = create_GAE_model(cfg_model, cfg_score, args.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(model, optimizer, data, splits, device, args.batch_size)
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f},  Cost Time: {time.time() - start:.4f}s')

        valid_loss = valid(model, data, splits, device, epoch)
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
    for model in ['GIN_Variant', 'GraphSAGE', 'Custom_GCN']:
        args.model = model
        for node_feature in ['random', 'original', 'adjacency', 'one-hot']:
            args.node_feature = node_feature

            for i in range(5):
                experiment_loop(args)