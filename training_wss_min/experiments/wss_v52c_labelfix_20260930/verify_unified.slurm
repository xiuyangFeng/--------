#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=v52c_verify
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/verify_unified_%j.out
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
/public/newhome/cy/.conda/envs/GNN/bin/python training_wss_min/experiments/wss_v52c_labelfix_20260930/verify_unified.py 2>&1 | grep -v "^\["
