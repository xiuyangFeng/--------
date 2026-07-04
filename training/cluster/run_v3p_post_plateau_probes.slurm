#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_k_probes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00

# V3P 后平台期 K1/K3/K5 CPU 短探针（node03）
set -eo pipefail

module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/f0_decision

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=== V3P post-plateau K probes $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_post_plateau_probes --verbose

echo ">>> post-plateau probes done $(date)"
