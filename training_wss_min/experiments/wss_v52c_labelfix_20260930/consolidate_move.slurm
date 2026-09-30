#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --job-name=v52c_move
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/consolidate_move_%j.out
# user-approved 2026-09-30: make v5.2c self-contained (moves only), then verify; purge is run separately after review.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd $G
export PYTHONPATH=$G
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/wss_v52c_labelfix_20260930
date -Is; hostname; sha1sum $E/consolidate_v52c.py | cut -c1-12
$PY $E/consolidate_v52c.py plan | tail -12
$PY $E/consolidate_v52c.py move
bash $E/verify_after_move.slurm
