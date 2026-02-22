#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=vivit_asterix_seeds
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

# One config, 8 seeds
SEEDS=(100 101 102 103 104 105 106 107)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd $SLURM_SUBMIT_DIR

# threading fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running seed=$SEED"

# Run your script
$PY_BIN train.py --algorithm AuxGazeFactorizedViViT --loading-method mine --gaze-loss-mode mean_then_kl --game Asterix --seed "$SEED"
