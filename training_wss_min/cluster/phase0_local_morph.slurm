#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=lmorph_phase0
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/wss_local_morph_20260917/offline/phase0d_advection.py
