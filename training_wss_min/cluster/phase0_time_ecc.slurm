#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=tecc_phase0
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
for s in d0_contract_audit d1_time_basis_ceiling d2_tnull d3_ecc_difficulty d4_asym_fidelity d5_physics_gate; do
  echo "===== $s $(date) ====="
  $PY -u $s.py 2>&1 | tee $s.txt
done
echo "===== done $(date) ====="
