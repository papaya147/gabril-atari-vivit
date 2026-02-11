#!/bin/bash
#SBATCH --job-name=vivit_no_gaze
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=shard:12

ALGOS=("FactorizedViViT" "AuxGazeFactorizedViViT")

# Task 0: FactorizedViViT, no gaze
# Task 1: AuxGazeFactorizedViViT, no gaze

ALGO_NAME=${ALGOS[$SLURM_ARRAY_TASK_ID]}

source /scr/adsrivat/gabril-atari-vivit/.venv/bin/activate

echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME (no gaze)"

srun python train.py --algorithm "$ALGO_NAME" --no-gaze
