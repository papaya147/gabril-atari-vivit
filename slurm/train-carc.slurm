#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|l40s"
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

ALGO_NAME=$1

if [ -z "$ALGO_NAME" ]; then
  echo "Error: No algorithm name provided."
  echo "Usage: sbatch script.sh <AlgorithmName>"
  exit 1
fi

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME"

# 4. Run your script
$PY_BIN train.py -- algorithm "$ALGO_NAME"