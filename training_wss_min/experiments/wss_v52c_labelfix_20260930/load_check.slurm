#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --job-name=v52c_loadchk
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/load_check_%j.out
# Strict loading of the new configs (IND / CV5 fold 2 / full265; train + test partitions) and every density level of the 9 units.
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export PYTHONPATH=$PWD OMP_NUM_THREADS=4
/public/newhome/cy/.conda/envs/GNN/bin/python training_wss_min/experiments/wss_v52c_labelfix_20260930/load_check.py 2>&1 | grep -v "^\["
