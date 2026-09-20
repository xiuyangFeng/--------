#!/bin/bash
#SBATCH --job-name=v5pilot
#SBATCH --partition=CPU
#SBATCH --time=0
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --output=outputs/wss_pinn/slurm/v5_pilot_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/v5_pilot_%j.err

set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN=${V5_RUN_NAME:-pilot_$(date +%Y%m%d)}
"$PY" -m wss_v5.build --pilot --workers "${V5_WORKERS:-12}" --run-name "$RUN" --domain-subsample "${V5_DOMAIN_SUBSAMPLE:-6000}" --n-generate "${V5_N_GENERATE:-60000}" ${V5_EXTRA_ARGS:-}
