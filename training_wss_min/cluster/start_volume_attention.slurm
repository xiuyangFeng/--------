#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=150G
#SBATCH --job-name=volume_formal
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
volume_smoke_state=${1:?Pass the completed 26-arm smoke queue_status.json}
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_volume_attention_queue --make-gate "$volume_smoke_state"
exec /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_volume_attention_queue --mode formal --slots-per-gpu 4
