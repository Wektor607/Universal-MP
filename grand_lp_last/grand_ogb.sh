#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=tun_collab

#SBATCH --output=log/Universal_MPNN_%j.output
#SBATCH --error=error/Universal_MPNN_%j.error
#SBATCH --account=hk-project-pai00023   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
#SBATCH --gres=gpu:1

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
# cd /hkfs/work/workspace/scratch/cc7738-rebuttal 
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP-German/Universal-MP/GRAND_LP_chen
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"

python tuning.py --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1 \
 --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.001 --prelr 0.001  --batch_size 65536 \
 --ln --lnnn --predictor cn1 --dataset collab  --epochs 40 --hidden_dim 128 --num_layers 3 --mplayers 1  \
 --testbs 131072  --device 0 --maskinput --res  --use_xlin  --tailact 

# Print loaded modules

# echo "Start time: $(date)"

# data_name=(ppa) #ppa citation2 

# for data in $data_name; do
# echo "Start training grand on $data"
# python main_grand.py  --dataset ogbl-$data_name --device 0 --no_early --beltrami
# # done
# python main_grand.py  --dataset ogbl-citation2 --device 0 --no_early --beltrami
# python main_grand.py  --dataset ogbl-collab --device 0 --no_early --beltrami

# python allin_grand.py  --data_name ogbl-collab --device 0 --no_early --beltrami --epoch 1000
# python allin_grand_original.py  --data_name ogbl-collab --device 0 --no_early --beltrami --epoch 1000
# python allin_grand_original.py  --data_name ogbl-collab --device 1 --gcn True --epoch 1000

# to sum up 

# for grand
# `python allin_grand_original.py  --data_name ogbl-collab --device 0 --no_early --beltrami --epoch 1000``

# for gcn
# `python allin_grand_original.py  --data_name ogbl-collab --device 1 --gcn True --epoch 1000`

# for comparison with the same hidden channels
# `
# python gnn_ogb_heart.py --data_name ogbl-collab --gnn_model GCN --hidden_channels 18 --lr 0.001 --dropout 0.0 --num_layers 3 --num_layers_predictor 3 --epochs 800 --kill_cnt 100 --batch_size 16394 --runs 2 --device 2
`