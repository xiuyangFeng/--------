#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=wss_new_v4
#SBATCH --output=pipeline_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=pipeline_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-170%4

set -euo pipefail

cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python \
    -m pipeline_wss_min.new_cohorts.preprocess --index "${SLURM_ARRAY_TASK_ID}"
