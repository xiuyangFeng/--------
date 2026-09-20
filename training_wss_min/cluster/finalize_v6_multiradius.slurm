#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=ms_report
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=00:30:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
"$PY" -u -m training_wss_min.tools.report_v6_multiradius --update-tracker
"$PY" -u -m training_wss_min.tools.update_v6_multiradius_xlsx
"$PY" -u -m training_wss_min.tools.plot_v6_multiradius
