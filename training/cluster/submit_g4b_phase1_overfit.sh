#!/usr/bin/env bash
# G4-b Phase 1 · 分支级 2D 展开 · 单 case 过拟合
# 用法（仓库根）: bash training/cluster/submit_g4b_phase1_overfit.sh
# 可选: CASE=slow/OTHER bash training/cluster/submit_g4b_phase1_overfit.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p training/cluster/logs

CASE="${CASE:-slow/LI_HUAN_GE}"
SEED="${SEED:-1}"
export G4_UNWRAP_MODE=branch
export G4_EXP_ID=V3P-G-61-2DUnwrapBranch
export G4_EPOCHS="${G4_EPOCHS:-200}"
export G4_LR="${G4_LR:-1e-3}"
export G4_BATCH="${G4_BATCH:-16}"

echo "=== G4-b Phase 1 · branch 单 case 过拟合 ==="
echo "case: $CASE · seed: $SEED · unwrap: branch"
echo "Go: grid R²≥0.95 且 remap gap<0.02"
echo ""

JID=$(sbatch --chdir="$ROOT" --job-name="g4b_of1c" \
    training/cluster/run_g4_2d_unwrap.slurm \
    overfit1c "$CASE" "$SEED" \
    | awk '{print $NF}')

echo "G4-b overfit1c -> Job ${JID}"
echo "  tail -f training/cluster/logs/g4b_of1c_${JID}.out"
