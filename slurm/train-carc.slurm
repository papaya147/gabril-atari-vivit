#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-15
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|l40s|v100|a40"
#SBATCH --mem=24G
#SBATCH --time=24:00:00

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")
SEEDS=(100 101 102 103 104 105 106 107)

# Game passed as first argument: sbatch slurm/train-carc.slurm Alien
GAME=${1:-ChopperCommand}

# 16 tasks = 2 algos x 8 seeds
# Task layout: ALGO_IDX = task/8, SEED_IDX = task%8

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 8))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))

ALGO_NAME=${ALGOS[$ALGO_IDX]}
LOADING_METHOD=gabril
GAZE_LOSS_MODE=kl_then_mean
SEED=${SEEDS[$SEED_IDX]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME with Loading Method: $LOADING_METHOD (gaze_loss_mode=$GAZE_LOSS_MODE) seed=$SEED on Game: $GAME"

# Run your script
$PY_BIN train.py --algorithm "$ALGO_NAME" --loading-method "$LOADING_METHOD" --gaze-loss-mode "$GAZE_LOSS_MODE" --game "$GAME" --seed "$SEED"