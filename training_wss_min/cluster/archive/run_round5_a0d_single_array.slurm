#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=wss_r5_a0d_single
#SBATCH --array=0-3%4
#SBATCH --output=training_wss_min/runs/_round5/a0d_fit_chain/simple_overfit/slurm_%A_%a.out
#SBATCH --error=training_wss_min/runs/_round5/a0d_fit_chain/simple_overfit/slurm_%A_%a.err
#SBATCH --time=00:30:00

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python3.10 -m training_wss_min.experiments.a0d_simple_overfit \
  --mode single --case-index "${SLURM_ARRAY_TASK_ID}" \
  --protocol training_wss_min/configs/round5/a0d_simple_overfit_protocol.json
