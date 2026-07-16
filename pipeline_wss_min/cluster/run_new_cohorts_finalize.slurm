#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=wss_new_qa
#SBATCH --output=pipeline_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=pipeline_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G

set -euo pipefail

cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python \
    -m pipeline_wss_min.new_cohorts.preprocess --summarize
/public/newhome/cy/.conda/envs/GNN/bin/python \
    -m pipeline_wss_min.new_cohorts.qa
