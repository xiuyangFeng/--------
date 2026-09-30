#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=wss_v52phys_prep
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
# v5.2 retraining (2026-09-23): splits (IND train170/test91, CV5 261 patient-grouped), partition WSS stats + feature z-scores, 40 arm configs, matrix.json.
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN_v52s_frozen_20260927
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_wss_v52_syn --children /public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/synth_children_ok59.txt
date -Is
