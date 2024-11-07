# Heuristic Learning Graph Neural Network (HL-GNN)

This repository contains the official implementation of Universal-MPNN, as presented in the paper [""]().

## Overview

HL-GNN is a novel method for link prediction that .... It demonstrates effectiveness in link prediction tasks and scales well to large OGB datasets. Notably, HL-GNN requires only a few parameters for training (excluding the predictor) and is significantly ...

<div align=center>
    <img src="/..png" alt="HL-GNN" width="60%" height="60%">
</div>

For more details, please refer to the []().

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ChenS676/Universal-MP
cd Universal-MP
pip install -r requirements.txt
```

## Usage

### Planetoid Datasets

#### Cora

```bash
cd Planetoid
python planetoid.py --dataset cora --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

#### Citeseer

```bash
cd Planetoid
python planetoid.py --dataset citeseer --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

#### Pubmed

```bash
cd Planetoid
python planetoid.py --dataset pubmed --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
```

### Amazon Datasets

#### Photo

```bash
cd Planetoid
python amazon.py --dataset photo --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

#### Computers

```bash
cd Planetoid
python amazon.py --dataset computers --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

### OGB Datasets

#### ogbl-collab

```bash
cd OGB
python main.py --data_name ogbl-collab --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
```

#### ogbl-ddi

```bash
cd OGB
python main.py --data_name ogbl-ddi --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
```

#### ogbl-ppa

```bash
cd OGB
python main.py --data_name ogbl-ppa --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 500 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
```

#### ogbl-citation2

```bash
cd OGB
python main.py --data_name ogbl-citation2 --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 100 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
```

## Results

The performance of HL-GNN on various datasets is summarized in the table below. The best and second-best performances are highlighted in **bold** and *italic*, respectively.

|         |   Cora    | Citeseer  |  Pubmed   |   Photo   | Computers |  collab   |    ddi    |    ppa    | citation2 |
| :-----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Method  | Hits@100  | Hits@100  | Hits@100  |    AUC    |    AUC    |  Hits@50  |  Hits@20  | Hits@100  |    MRR    |
|  SEAL   |           |           |           |           |           |           |           |           |           |
| NBFNet  |           |           |           |           |           |           |           |           |           |
| Neo-GNN |           |           |           |           |           |           |           |           |           |
|  BUDDY  |           |           |           |           |           |           |           |           |           |
| HL-GNN  |           |           |           |           |           |           |           |           |           |


## Acknowledgement

We sincerely thank the [PLNLP repository](https://github.com/zhitao-wang/PLNLP) for providing an excellent pipeline that greatly facilitated our work on the OGB datasets.

## Citation

If you find HL-GNN useful in your research, please cite our paper:

```bibtex
```

Feel free to reach out if you have any questions!

