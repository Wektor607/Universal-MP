
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import csv
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt


from yacs.config import CfgNode

from baselines.MLP import MLPPolynomialFeatures
from baselines.utils import loaddataset
from baselines.heuristic import AA, RA, Ben_PPR, katz_apro, shortest_path
from baselines.heuristic import CN as CommonNeighbor
from baselines.GNN import (
    GAT_Variant, GCN_Variant, SAGE_Variant, GIN_Variant, 
    GAE_forall, InnerProduct, mlp_score
)
from yacs.config import CfgNode as CN
from archiv.mlp_heuristic_main import EarlyStopping
from baselines.LINKX import LINKX


class Config:
    def __init__(self):
        self.epochs = 100
        self.dataset = "ddi"
        self.batch_size = 8192
        self.h_key = "PPR"
        self.gnn = "gcn"
        self.model = "LINKX"
        self.use_feature = False
        self.node_feature = 'original'  # 'one-hot', 'random', 
                                        # 'quasi-orthogonal'
        self.use_early_stopping = True


def init_LINKX(
    cfg_model: CN,
    cfg_score: CN,
    m_name: str):
    
    if m_name in ['LINKX']:
        encoder = LINKX(
            num_nodes=cfg_model.num_nodes,
            in_channels=cfg_model.in_channels,
            hidden_channels=cfg_model.hidden_channels,
            out_channels=cfg_model.out_channels,
            num_layers=cfg_model.num_layers,
            num_edge_layers=cfg_model.num_edge_layers,
            num_node_layers=cfg_model.num_node_layers,
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
        # Without this else I got: UnboundLocalError: local variable 'model'
        # referenced before assignment
        raise ValueError('Current model does not exist')

    return GAE_forall(encoder=encoder, decoder=decoder)


def train(model, optimizer, data, splits, device, batch_size=512):
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

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )

        z = model.encode(
            data.x, batch_edge_index
        )

        pos_pred = model.decode(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
        ).squeeze()
        neg_pred = model.decode(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
        ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        batch_loss.backward()

        total_loss += batch_loss.item()

        optimizer.step()

    return total_loss / num_batches


@torch.no_grad()
def valid(model, data, splits, device, batch_size=512):
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

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )

        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
            ).squeeze()
        neg_pred = model.decode(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
            ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model, data, splits, device, batch_size=512):
    model.eval()

    # Positive and negative edges for test
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

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )
        
        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
            ).squeeze()
        neg_pred = model.decode(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
            ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches


def save_to_csv(file_path,
                model_name,
                node_feat,
                heuristic,
                test_loss):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model', 'NodeFeat', 'Heuristic', 'Test_Loss'])
        writer.writerow([model_name, node_feat, heuristic, test_loss])
    print(f'Saved {model_name, node_feat, heuristic, test_loss} to '
          f'{file_path}')


def visualize(pred, true_label, save_path='./visualization.png'):
    pred = pred.cpu().detach().numpy()
    true_label = true_label.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(true_label)), true_label, color='#A6CEE3',
                label='True Score',
                alpha=0.6)
    plt.scatter(np.arange(len(pred)), pred, color='#B2DF8A',
                label='Prediction', alpha=0.6)

    plt.title('Predictions vs True Score')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.ylim(0, 1.5)
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    print(f"Visualization saved at {save_path}")


def experiment_loop(args: Config, num_experiments=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and set up data
    data, splits = loaddataset(args.dataset, None)
    data = data.to(device)

    edge_weight = torch.ones(data.edge_index.size(1), dtype=float)
    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.num_nodes, data.num_nodes)
    )

    # Configure node features based on specified type
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
    with open('./yamls/cora/heart_gnn_models.yaml', "r") as f:
        cfg = CfgNode.load_cfg(f)

    cfg_model = eval(f'cfg.model.{args.model}')
    cfg_score = eval(f'cfg.score.{args.model}')
    cfg.model.type = args.model

    if not hasattr(splits['train'], 'x') or splits['train'].x is None:
        cfg_model.in_channels = 1024
    else:
        cfg_model.in_channels = data.x.size(1)

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

    # Store results of each experiment
    early_stopping = EarlyStopping(patience=5, verbose=True)

    if args.model == 'LINKX':
        cfg_model.in_channels = data.x.size(-1)
        cfg_model.num_nodes = data.num_nodes
        model = init_LINKX(cfg_model, cfg_score, 'LINKX').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(
            model, optimizer, data, splits, device, args.batch_size
        )

        # Validate every 20 epochs
        if epoch % 20 == 0:
            valid_loss = valid(model, data, splits, device)
            print(
                f'Epoch: {epoch:03d}, train loss: {train_loss:.4f}, '
                f'valid loss: {valid_loss:.4f}, '
                f'Cost Time: {time.time() - start:.4f}s')

            if args.use_early_stopping:
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Training stopped early!")
                    break

    test_loss = test(model, data, splits, device)

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