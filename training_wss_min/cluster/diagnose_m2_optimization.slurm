#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=m2_diagnostics
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
/public/newhome/cy/.conda/envs/GNN/bin/python -u - <<'PY'
import json
import math
from pathlib import Path
import time
import torch
from training_wss_min.tools.m2_optimization_common import EXP, RUNS, PREFIX
assert torch.cuda.device_count() == 1
preflight = json.loads((EXP / "runtime_preflight.json").read_text())
assert preflight["passed"]
peaks = [arm["max_memory_reserved_mib"] for arm in preflight["arms"].values()]
peaks.extend(json.loads(path.read_text())["peak_cuda_reserved_mb"]
             for path in (RUNS / PREFIX).glob("MO*_s1234/training_diagnostics.json"))
assert peaks and all(math.isfinite(value) and value > 0 for value in peaks)
required = max(7000., 1.25 * max(peaks) + 2048.)
while True:
    available = torch.cuda.mem_get_info()[0] / 2**20
    if available >= required:
        print(f"GPU memory gate passed: free={available:.0f} MiB, required={required:.0f} MiB", flush=True)
        break
    print(f"Waiting for memory: free={available:.0f} MiB, required={required:.0f} MiB", flush=True)
    time.sleep(15)
PY
exec /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.diagnose_m2_optimization "$@"
