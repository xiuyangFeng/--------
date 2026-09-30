#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
export GNN_JOINT_SOURCE_ROOT=/public/newhome/cy/Digital_twin/GNN
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52_20260929/frozen/f0_s1234_v1
/public/newhome/cy/.conda/envs/GNN/bin/python -u report.py --config-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52_20260929/frozen/f0_s1234_v1/configs --experiment-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52_20260929
