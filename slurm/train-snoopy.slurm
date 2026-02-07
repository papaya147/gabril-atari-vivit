#!/bin/bash
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=shard:12

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")
METHODS=("mine" "gabril")

# Task 0: FactorizedViViT, mine
# Task 1: FactorizedViViT, gabril
# Task 2: AuxGazeFactorizedViViT, mine
# Task 3: AuxGazeFactorizedViViT, gabril

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 2))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % 2))

ALGO_NAME=${ALGOS[$ALGO_IDX]}
LOADING_METHOD=${METHODS[$METHOD_IDX]}

source /scr/adsrivat/gabril-atari-vivit/.venv/bin/activate

echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME with Loading Method: $LOADING_METHOD"

srun python train.py --algorithm "$ALGO_NAME" --loading_method "$LOADING_METHOD"