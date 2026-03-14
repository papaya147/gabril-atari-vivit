#!/bin/bash
#SBATCH --job-name=vivit_no_gaze
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-15%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=shard:12
#SBATCH --mem=24G
#SBATCH --time=24:00:00

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")
SEEDS=(100 101 102 103 104 105 106 107)

# Game passed as first argument: sbatch slurm/train-snoopy-no-gaze.slurm ChopperCommand
GAME=${1:-ChopperCommand}

# 16 tasks = 2 algos x 8 seeds
# Task layout: ALGO_IDX = task/8, SEED_IDX = task%8

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / 8))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 8))

ALGO_NAME=${ALGOS[$ALGO_IDX]}
SEED=${SEEDS[$SEED_IDX]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/scr/adsrivat/gabril-atari-vivit/.venv/bin/activate

echo "Current directory: $(pwd)"
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME (no gaze) seed=$SEED on Game: $GAME"

$PY_BIN train.py --algorithm "$ALGO_NAME" --no-gaze --game "$GAME" --seed "$SEED"
