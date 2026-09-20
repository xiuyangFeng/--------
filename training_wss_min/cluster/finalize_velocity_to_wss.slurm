#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=velwss_report
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=00:30:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
if [[ "${1:-}" != "--workbook-only" ]]; then
    /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.verify_velocity_to_wss
    /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.report_velocity_to_wss
fi
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.update_velocity_to_wss_xlsx
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.update_velocity_to_wss_xlsx --verify-only
