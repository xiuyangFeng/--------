#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=tecc_prepare
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_wss_time_ecc
