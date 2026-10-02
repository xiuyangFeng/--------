#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --job-name=v52d_load
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/load_check_%j.out
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"; export PYTHONPATH="$G" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; /public/newhome/cy/.conda/envs/GNN/bin/python training_wss_min/experiments/wss_v52d_merge_20261001/08_load_check.py; date -Is
