#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=vivit_no_gaze
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|l40s|v100|a40"
#SBATCH --mem=24G
#SBATCH --time=24:00:00

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")

ALGO_NAME=${ALGOS[$SLURM_ARRAY_TASK_ID]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME (no gaze)"

# Run your script
$PY_BIN train.py --algorithm "$ALGO_NAME" --no-gaze
