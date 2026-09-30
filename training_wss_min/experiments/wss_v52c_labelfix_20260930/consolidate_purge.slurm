#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=v52c_purge
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/consolidate_purge_%j.out
# user-approved 2026-09-30 (class A): delete the superseded roots / caches after move + verification (verify_after_move_16384: passed)
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd $G
export PYTHONPATH=$G
date -Is; hostname
/public/newhome/cy/.conda/envs/GNN/bin/python $G/training_wss_min/experiments/wss_v52c_labelfix_20260930/consolidate_v52c.py purge --yes
date -Is
