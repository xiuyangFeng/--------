#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_g5_cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6:00:00

# G5 并行 CPU：I6-diag L2/L3 eval + postview 三联图（三例）
set -eo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$TRAINING_ENV" 2>/dev/null || true
fi
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/field/plots/stl_surface_compare

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

I6_RUN="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
MANIFEST="${I6_RUN}/predictions_test_best_wss/manifest.json"
RUN_TAG="v3p_i6diag_t016_report"

echo "=== [1/2] I6-diag L2/L3 eval $(date) ==="
"$PYTHON_BIN" -m training.scripts.evaluate_field_run_full \
    --run-dir "$I6_RUN" \
    --checkpoint best_wss_model.pt \
    --subset test \
    --skip-predict \
    --clinical-pa

echo "=== [2/2] postview WSS/P 三联图（三例） $(date) ==="
for CASE in slow/GUO_XI_JIANG slow/ZHANG_JUN_HUA fast/CHEN_SHI_MING; do
    echo "--- CASE ${CASE} ---"
    MANIFEST="$MANIFEST" \
    CASE_NAME="$CASE" \
    SAMPLE_ID=result_features_merged-1146 \
    RUN_TAG="$RUN_TAG" \
    bash tools/cfdpost_cloud_export/run_case_surface_compare.sh
done

echo ">>> G5 CPU 并行任务完成 $(date)"
