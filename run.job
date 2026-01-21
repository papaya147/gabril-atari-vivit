#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x-%A_%a.out

cd $SLURM_SUBMIT_DIR

# 1. Threading Fix
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Define the EXACT path to your Python binary
# (We bypass 'conda activate' entirely to avoid shell errors)
export PY_BIN=/home1/adsrivat/.conda/envs/.venv/bin/python

# 3. Sanity Checks
echo "Job started on $(hostname) at $(date)"
nvidia-smi

echo "------------------------------------------------------"
echo "Using Python binary at: $PY_BIN"
$PY_BIN -c "import torch; print(f'Torch version: {torch.__version__}')"
echo "------------------------------------------------------"

# 4. Run your script using the specific binary
$PY_BIN train.py $SLURM_ARRAY_TASK_ID