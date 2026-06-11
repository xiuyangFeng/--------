#!/usr/bin/env bash
# G4 Phase 2 · 全量 split Probe（正式训练 · 1×GPU）
# 用法（仓库根）: bash training/cluster/submit_g4_phase2_probe.sh
#
# 对照: post5463 band 0.425±0.012 · Go: test grid R² Δ≥+0.03（相对 0.425）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p training/cluster/logs

echo "=== G4 Phase 2 · 2D U-Net Probe（正式 SLURM 提交）==="
echo "工作目录: $ROOT"
echo "Phase 1 过拟合: slow/LI_HUAN_GE grid R²=0.954 ✅（登录节点探针，非本作业）"
echo ""

# 可选 seed：默认先 seed1；第二参数传 2 则提交 seed2
SEED="${1:-1}"
export G4_EPOCHS="${G4_EPOCHS:-150}"
export G4_LR="${G4_LR:-1e-3}"
export G4_BATCH="${G4_BATCH:-32}"
export G4_EXP_ID="V3P-G-60-2DUnwrap"

JID=$(sbatch --job-name="g4_probe_s${SEED}" \
    training/cluster/run_g4_2d_unwrap.slurm \
    probe "-" "$SEED" \
    | awk '{print $NF}')

echo "G4 Probe seed${SEED} -> Job ${JID}"
echo ""
echo "监控:"
echo "  squeue -u \$USER"
echo "  tail -f training/cluster/logs/g4_probe_s${SEED}_${JID}.out"
echo ""
echo "对照 band: 5466/5468/5478 wss 0.425±0.012 · checkpoint 按 val_r2_wss_grid_phys 选优"
