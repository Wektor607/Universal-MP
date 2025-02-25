#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --partition=accelerated
#SBATCH --job-name=gcn4cn
#SBATCH --gres=gpu:1

#SBATCH --output=log/UniversalMPNN_%j.output
#SBATCH --error=error/UniversalMPNN_%j.error
#SBATCH --account=hk-project-pai00001  # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/trials

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --output=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/
#SBATCH --error=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/error
#SBATCH --job-name=exp1


# execute your commands
cd /pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/

# Array of model names
models=("Custom_GAT" "Custom_GIN") #"LINKX"  "Custom_GCN" "GraphSAGE"
nodefeat=("adjacency"  "one-hot" "random" "original")

for model in "${models[@]}"; do
    for nodefeat in "${nodefeat[@]}"; do
        echo "python gcn2struc.py --model "$model" --dataset "ddi" --nodefeat "$nodefeat" --h_key "CN" "
        python gcn2struc.py --model "$model" --dataset  "ddi"  --nodefeat "$nodefeat" --h_key "CN"
    done
done 


wait

