import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os.path as osp
import argparse

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import time
import pickle
from data import get_dataset, my_get_dataset


def main(opt):
    dataset_name = opt['dataset']

    print(f"[i] Generating embeddings for dataset: {dataset_name}")
    if dataset_name in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2']:
        data, split_edge = my_get_dataset('/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-German/Universal-MP/GRAND_LP/pos_enc_genereation/dataset', opt, dataset_name)
    else:
        dataset = get_dataset(opt, '/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-German/Universal-MP/GRAND_LP/pos_enc_genereation/dataset')
        data = dataset.data
    
    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(data.edge_index, embedding_dim=opt['embedding_dim'], walk_length=opt['walk_length'],
                     context_size=opt['context_size'], walks_per_node=opt['walks_per_node'],
                     num_negative_samples=opt['neg_pos_ratio'], p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        if dataset_name in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2']:
            if dataset_name == 'ogbl-citation2':
                source_node = split_edge['test']['source_node']
                target_node = split_edge['test']['target_node']
                target_node_neg = split_edge['test']['target_node_neg']

                # Положительные рёбра (shape: [N, 2])
                pos_test_edge = torch.stack([source_node, target_node], dim=1)

                # Отрицательные рёбра (shape: [N*K, 2])
                # Если у каждой позиции N по K негативных целей, разворачиваем их:
                src_expanded = source_node.unsqueeze(1).expand(-1, target_node_neg.size(1))
                neg_test_edge = torch.stack([src_expanded, target_node_neg], dim=2)
                neg_test_edge = neg_test_edge.view(-1, 2)
            else:
                pos_test_edge = split_edge['test']['edge']
                neg_test_edge = split_edge['test']['edge_neg']

            # Функция для оценки точности (AUC)
            def compute_auc(pos_edge, neg_edge):
                pos_score = (z[pos_edge[:, 0]] * z[pos_edge[:, 1]]).sum(dim=-1).sigmoid()
                if dataset_name == 'ogbl-citation2':
                    neg_scores = []
                    neg_labels = torch.zeros(neg_edge.shape[0], device=z.device)
                    batch_size = 2048
                    for i in range(0, neg_edge.shape[0], batch_size):
                        batch = neg_edge[i : i + batch_size]
                        neg_score = (z[batch[:, 0]] * z[batch[:, 1]]).sum(dim=-1).sigmoid()
                        neg_scores.append(neg_score)

                    neg_scores = torch.cat(neg_scores)
                else:
                    neg_score = (z[neg_edge[:, 0]] * z[neg_edge[:, 1]]).sum(dim=-1).sigmoid()

                pos_labels = torch.ones(pos_score.size(0))
                neg_labels = torch.zeros(neg_score.size(0))

                labels = torch.cat([pos_labels, neg_labels], dim=0)
                scores = torch.cat([pos_score, neg_score], dim=0)

                # Рассчитываем AUC
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(labels.cpu(), scores.cpu())

            # Оцениваем AUC на тестовом наборе
            acc = compute_auc(pos_test_edge, neg_test_edge)
        else:
            acc = model.test(z[data.train_mask], data.y[data.train_mask],
                            z[data.test_mask], data.y[data.test_mask],
                            max_iter=150)
        return acc, z

    
    def log_memory():
        print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    ### here be main code
    import gc
    t = time.time()
    for epoch in range(1, opt['epochs']+1):
        log_memory()
        loss = train()
        train_t = time.time() - t
        t = time.time()
        acc, _ = test()
        test_t = time.time() - t
        print(f'Epoch: {epoch:02d}, Train: {train_t:.2f}, Test: {test_t:.2f},  Loss: {loss:.4f}, Acc: {acc:.4f}')
        log_memory()
        torch.cuda.empty_cache()
        gc.collect()

    acc, z = test()
    print(f"[i] Final accuracy is {acc}")
    print(f"[i] Embedding shape is {z.data.shape}")

    fname = "DW_%s_emb_%03d_wl_%03d_cs_%02d_wn_%02d_epochs_%03d.pickle" % (
      opt['dataset'], opt['embedding_dim'], opt['walk_length'], opt['context_size'], opt['walks_per_node'], opt['epochs']
    )

    save_path = osp.join("/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-German/Universal-MP/GRAND_LP/pos_enc_genereation/dataset/pos_encodings")

    # Создаем директорию, если её нет
    os.makedirs(save_path, exist_ok=True)
    print(f"[i] Storing embeddings in {fname}")
    
    with open(osp.join(save_path, fname), 'wb') as f:
      # make sure the pickle is not bound to any gpu, and store test acc with data
      pickle.dump({"data": z.data.to(torch.device("cpu")), "acc": acc}, f)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
  parser.add_argument('--walk_length', type=int, default=20, # note this can grow much bigger (paper: 40~100)
                        help='Walk length')
  parser.add_argument('--context_size', type=int, default=16,# paper shows increased perf until 16
                        help='Context size')
  parser.add_argument('--walks_per_node', type=int, default=16, # best paper results with 18
                        help='Walks per node')
  parser.add_argument('--neg_pos_ratio', type=int, default=1, 
                        help='Number of negatives for each positive')
  parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
  parser.add_argument('--gpu', type=int, default=2, 
                        help='GPU id (default 0)')
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")


  args = parser.parse_args()
  opt = vars(args)
  opt['rewiring'] = None
  main(opt)
