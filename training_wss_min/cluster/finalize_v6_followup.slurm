#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=v6_followup_report
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
"$PY" -u -m training_wss_min.tools.report_v6_followup --update-tracker
"$PY" -u -m training_wss_min.tools.update_v6_followup_xlsx
"$PY" -u -m training_wss_min.tools.plot_v6_followup
