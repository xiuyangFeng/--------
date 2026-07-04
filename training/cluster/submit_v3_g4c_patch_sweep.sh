#!/usr/bin/env bash
# V3P G4-c Phase 0 参数扫（并行 CPU · node03）
# 对照工程方案 §0.1：patch 12/16 · radius 4.5/6 · knn 20
# 基线 16x16/r3/k12 已由 v3p_g4c_patch_feasibility_20260629.json 覆盖，此处不重复。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/field/f0_decision

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  else
    echo "sbatch 不可用" >&2
    exit 1
  fi
fi

DATE_TAG="$(date +%Y%m%d)"
submit_one() {
  local ph=$1 pw=$2 radius=$3 knn=$4
  local tag="p${ph}_r${radius/./}_k${knn}"
  local out="outputs/field/f0_decision/v3p_g4c_patch_feasibility_${DATE_TAG}_${tag}.json"
  local job_id
  job_id=$("$SBATCH_BIN" --chdir="$ROOT" --job-name="g4c_${tag}" \
    training/cluster/run_v3_g4c_patch_feasibility.slurm \
    --patch-h "$ph" --patch-w "$pw" \
    --geodesic-radius-mm "$radius" --knn-k "$knn" \
    --output "$out" 2>&1 | awk '{print $NF}')
  echo "Job ${job_id} · ${tag} → ${out}"
}

echo "=== G4-c Phase 0 sweep (${DATE_TAG}) ==="
# patch 12x12
submit_one 12 12 3.0 20
submit_one 12 12 4.5 12
submit_one 12 12 4.5 20
submit_one 12 12 6.0 12
submit_one 12 12 6.0 20
# patch 16x16 · radius/knn 扩展
submit_one 16 16 4.5 12
submit_one 16 16 4.5 20
submit_one 16 16 6.0 12
submit_one 16 16 6.0 20

echo "=== 共 9 个 sweep job 已提交（不含已跑基线 16/3.0/12）==="
