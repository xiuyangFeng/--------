#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_p1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00
set -eo pipefail
module purge
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
J4_RUN="outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
echo "=== Phase 1 active selection $(date) ==="
"$PYTHON_BIN" -m training.scripts.run_v3p_phase1_active_data \
  --reference-run "$I6_RUN" --compare-run "$J4_RUN" \
  --date "$DATE_TAG" --top-n 15 --top-n-draft 10
echo ">>> Phase1 done $(date)"
