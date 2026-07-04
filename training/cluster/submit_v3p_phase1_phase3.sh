#!/usr/bin/env bash
# 提交 Phase 1 + Phase 3：J4 GPU 推理 → CPU 主动选数 + L3 eval
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

export PATH="/public/slurm/bin:/public/slurm/sbin:${PATH:-}"

J4_RUN="outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
MANIFEST="${J4_RUN}/predictions_test_best_wss/manifest.json"

mkdir -p training/cluster/logs logs

if [[ -f "$MANIFEST" ]]; then
  echo "J4 predictions 已存在，仅提交 Phase1+Phase3 CPU"
  JOB_P13=$(sbatch training/cluster/run_v3p_phase1_phase3.slurm | awk '{print $4}')
  echo "Phase1+Phase3 Job: $JOB_P13"
  squeue -u "${USER:-cy}" -j "$JOB_P13" 2>/dev/null || true
  exit 0
fi
JOB_PRED=$(sbatch training/cluster/run_v3p_j4_predict.slurm | awk '{print $4}')
echo "J4 predict Job: $JOB_PRED"

JOB_P13=$(sbatch --dependency=afterok:"$JOB_PRED" training/cluster/run_v3p_phase1_phase3.slurm | awk '{print $4}')
echo "Phase1+Phase3 Job: $JOB_P13 (afterok $JOB_PRED)"

squeue -u "${USER:-cy}" 2>/dev/null || true
