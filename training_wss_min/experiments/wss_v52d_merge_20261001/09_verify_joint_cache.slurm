#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=jc_v52d_verify
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/joint_cache_verify_%j.out
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"; export PYTHONPATH="$G" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; sha1sum training_wss_min/joint_cycle_data.py | cut -c1-12
/public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/wss_v52d_merge_20261001/09_verify_joint_cache.py --workers 10
date -Is
