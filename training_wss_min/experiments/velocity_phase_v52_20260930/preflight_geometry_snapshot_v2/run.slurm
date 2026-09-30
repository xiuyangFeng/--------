#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=velphase_geocheck
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/preflight_geometry_snapshot_v2/slurm_%j.log
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/preflight_geometry_snapshot_v2/slurm_%j.err
set -euo pipefail
unset PYTHONPATH PYTHONHOME GNN_JOINT_SPLIT_PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GNN_JOINT_SOURCE_ROOT=/public/newhome/cy/Digital_twin/GNN
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/preflight_geometry_snapshot_v2
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.velocity_phase_data prepare --cache-root /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache --geometry-root /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/velocity_phase_v52_20260930/preflight_geometry --workers 1 --limit 1
