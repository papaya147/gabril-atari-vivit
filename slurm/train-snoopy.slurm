#!/bin/bash
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=shard:12

ALGO_NAME=$1

if [ -z "$ALGO_NAME" ]; then
  echo "Error: No algorithm name provided."
  echo "Usage: sbatch train-snoopy.slurm <AlgorithmName>"
  exit 1
fi

source /scr/adsrivat/gabril-atari-vivit/.venv/bin/activate

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Running Algorithm: $ALGO_NAME"

srun python train.py --algorithm "$ALGO_NAME"