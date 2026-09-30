#!/bin/bash
#SBATCH --job-name=velocity_completion_audit
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/audit_completion_%j.log
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/audit_completion_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/velocity_phase_v52_20260930/analysis_20260930/audit_completion.py
