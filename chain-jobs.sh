#!/bin/bash

# Name of your slurm script
JOB_SCRIPT="train.slurm"

# How many jobs to chain
NUM_JOBS=3

echo "Submitting Job 1..."
JOB_ID=$(sbatch --parsable $JOB_SCRIPT)
echo "Job 1 ID: $JOB_ID"

for i in $(seq 2 $NUM_JOBS); do
    echo "Submitting Job $i (dependent on $JOB_ID)..."
    JOB_ID=$(sbatch --parsable --dependency=afterany:$JOB_ID $JOB_SCRIPT)
    echo "Job $i ID: $JOB_ID"
done