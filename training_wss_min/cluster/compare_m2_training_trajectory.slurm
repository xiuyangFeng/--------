#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=m2_trajectory
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=00:30:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
/public/newhome/cy/.conda/envs/GNN/bin/python -u - <<'PY'
import time
import torch
assert torch.cuda.device_count() == 1
while True:
    available = torch.cuda.mem_get_info()[0] / 2**20
    if available >= 7000:
        print(f"Allocated GPU has {available:.0f} MiB free; starting read-only training diagnosis.", flush=True)
        break
    print(f"Waiting for free memory: {available:.0f} MiB", flush=True)
    time.sleep(15)
PY
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.compare_m2_training_trajectory --steps 12
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.compare_m2_training_trajectory --steps 12 --fp32
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.diagnose_m2_optimization --smoke
