#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|l40s|v100|a40"
#SBATCH --mem=24G
#SBATCH --time=24:00:00

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")
METHODS=("mine" "gabril")
GAZE_LOSS_MODES=("mean_then_kl" "kl_then_mean")
# SEEDS=(100 101 102 103 104 105 106 107)

# 8 tasks = 2 algos x 2 methods x 2 gaze loss modes
# Task layout: ALGO_IDX = task/4, METHOD_IDX = (task%4)/2, GAZE_IDX = task%2

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 4))
METHOD_IDX=$(((SLURM_ARRAY_TASK_ID % 4) / 2))
GAZE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

ALGO_NAME=${ALGOS[$ALGO_IDX]}
LOADING_METHOD=${METHODS[$METHOD_IDX]}
GAZE_LOSS_MODE=${GAZE_LOSS_MODES[$GAZE_IDX]}
# SEED=${SEEDS[$SEED_IDX]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME with Loading Method: $LOADING_METHOD (gaze_loss_mode=$GAZE_LOSS_MODE)"

# Run your script
$PY_BIN train.py --algorithm "$ALGO_NAME" --loading-method "$LOADING_METHOD" --gaze-loss-mode "$GAZE_LOSS_MODE"