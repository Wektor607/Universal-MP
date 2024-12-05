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
#SBATCH --mail-user=chen.shao2@kit.edu

# Request GPU resources
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate ss

# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/trials


models=("Custom_GAT" "Custom_GCN" "GraphSAGE" "Custom_GIN" "LINKX")

for model in "${models[@]}"; do
    echo "python gcn2struc.py --model "$model" &"
    python gcn2struc.py --model "$model" --h_key PPR & 
done

wait


python gcn2struc.py --model "$model" --h_key PPR & 