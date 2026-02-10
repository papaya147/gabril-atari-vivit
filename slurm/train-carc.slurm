#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|l40s|v100|a40"
#SBATCH --mem=48G
#SBATCH --time=24:00:00

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")
METHODS=("mine" "gabril")
# SEEDS=(100 101 102 103 104 105 106 107)

# 32 tasks = 4 (algo x method) x 8 seeds
# Task layout: COMBO_IDX = task / 8, SEED_IDX = task % 8

# COMBO_IDX=$((SLURM_ARRAY_TASK_ID / 8))
# SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))

ALGO_IDX=$((COMBO_IDX / 2))
METHOD_IDX=$((COMBO_IDX % 2))

ALGO_NAME=${ALGOS[$ALGO_IDX]}
LOADING_METHOD=${METHODS[$METHOD_IDX]}
# SEED=${SEEDS[$SEED_IDX]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME with Loading Method: $LOADING_METHOD"

# Run your script
$PY_BIN train.py --algorithm "$ALGO_NAME" --loading-method "$LOADING_METHOD"