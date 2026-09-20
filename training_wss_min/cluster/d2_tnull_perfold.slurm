#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=tecc_d2
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline
export OMP_NUM_THREADS=8
/public/newhome/cy/.conda/envs/GNN/bin/python -u d2_tnull.py 2>&1 | tee d2_tnull.txt
