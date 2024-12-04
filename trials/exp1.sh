#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=single
#SBATCH --mail-type=ALL
#SBATCH --output=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/
#SBATCH --error=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/error
#SBATCH --job-name=exp1
#SBATCH --mem-per-cpu=


# execute your commands
cd /pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/

# Array of model names
models=("Custom_GAT" "Custom_GCN" "GraphSAGE" "Custom_GIN" "LINKX")

# Iterate over each model and run the Python script in the background
for model in "${models[@]}"; do
    echo "python gcn2struc.py --model "$model" &"
    python gcn2struc.py --model "$model" &
done

# Optional: Wait for all background processes to complete before exiting the script
wait

python gcn2struc.py --model Custom_GAT
python gcn2struc.py --model Custom_GCN
python gcn2struc.py --model GraphSAGE
python gcn2struc.py --model Custom_GIN
python gcn2struc.py --model LINKX
