#!/usr/bin/env bash
# G4-c Phase 1 · 双 case 单 case 过拟合（CHEN fast + GUO slow）
# 配置：Phase 0 领先 p12/r3/k20 · v3p_g4c_patch_sweep_summary_20260630.json
#
# 用法（仓库根）:
#   bash training/cluster/submit_v3_g4c_phase1_overfit.sh
# 可选:
#   CASES="fast/CHEN_SHI_MING" bash training/cluster/submit_v3_g4c_phase1_overfit.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  else
    echo "sbatch 不可用" >&2
    exit 1
  fi
fi

CASES="${CASES:-fast/CHEN_SHI_MING slow/GUO_XI_JIANG}"
SEED="${SEED:-1}"

export G4C_PATCH_H="${G4C_PATCH_H:-12}"
export G4C_PATCH_W="${G4C_PATCH_W:-12}"
export G4C_RADIUS_MM="${G4C_RADIUS_MM:-3.0}"
export G4C_KNN_K="${G4C_KNN_K:-20}"
export G4C_EPOCHS="${G4C_EPOCHS:-200}"
export G4C_LR="${G4C_LR:-1e-3}"
export G4C_BATCH="${G4C_BATCH:-32}"
export G4C_EXP_ID="${G4C_EXP_ID:-V3P-M-E-G4c-PatchPhase1-post5463}"

echo "=== G4-c Phase 1 提交 ==="
echo "patch ${G4C_PATCH_H}x${G4C_PATCH_W} · r=${G4C_RADIUS_MM}mm · k=${G4C_KNN_K}"
echo "Go: grid R²≥0.95 且 merged 3D R² − I6-diag ≥ +0.05"
echo ""

for CASE in $CASES; do
  slug="${CASE//\//__}"
  jid=$("$SBATCH_BIN" --chdir="$ROOT" --job-name="g4c_p1_${slug}" \
    --export=ALL \
    training/cluster/run_g4c_patch_overfit.slurm \
    overfit1c "$CASE" "$SEED" 2>&1 | awk '{print $NF}')
  echo "Job ${jid} · ${CASE} · tail -f logs/g4c_p1_${slug}_${jid}.out"
done

echo ""
echo "判读：run_dir/summary.json · go_overfit=true 才考虑全量训练"
