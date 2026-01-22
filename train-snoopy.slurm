#!/bin/bash
#SBATCH --job-name=vivit_v2=3_train
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# --- RESOURCE REQUESTS ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# --- SHARDING CONFIGURATION ---
#SBATCH --gres=gpu:1

# --- THE ARRAY CONTROLLER ---
#SBATCH --array=0-14%4

# --- SETUP ---
source /scr/adsrivat/gabril-atari-vivit/.venv/bin/activate

echo pwd
echo "Job started on $(hostname) at $(date)"
echo "Assigned GPU resources:"
nvidia-smi

srun python train.py $SLURM_ARRAY_TASK_ID