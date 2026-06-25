#!/usr/bin/env bash
# V3P M-E 并行探针：WSS-only + 弱压力（与 I6-a 正交 · I7 feature_decay 互补）
# 用法（仓库根）:
#   bash training/cluster/submit_v3_me_parallel_probes.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  else
    echo "错误: 找不到 sbatch"
    exit 1
  fi
fi

SLURM="$ROOT/training/cluster/run_train_field.slurm"
CFG_WSS="$ROOT/training/configs/field/generated/v3_pointcloud/V3P-M-E-WSSOnly-AsymW-a-post5463_seed1.json"
CFG_PWEAK="$ROOT/training/configs/field/generated/v3_pointcloud/V3P-M-E-PWeak-AsymW-a-post5463_seed1.json"

for cfg in "$CFG_WSS" "$CFG_PWEAK"; do
  if [ ! -f "$cfg" ]; then
    echo "错误: 配置文件不存在: $cfg"
    exit 1
  fi
done

job_wss=$("$SBATCH_BIN" -p GPU --gres=gpu:1 --job-name="v3p_me_wssonly" \
  --chdir="$ROOT" \
  --output="$ROOT/logs/v3p_me_wssonly_%j.out" \
  --error="$ROOT/logs/v3p_me_wssonly_%j.err" \
  "$SLURM" \
  "$CFG_WSS" \
  | awk '{print $NF}')

job_pweak=$("$SBATCH_BIN" -p GPU --gres=gpu:1 --job-name="v3p_me_pweak" \
  --chdir="$ROOT" \
  --output="$ROOT/logs/v3p_me_pweak_%j.out" \
  --error="$ROOT/logs/v3p_me_pweak_%j.err" \
  "$SLURM" \
  "$CFG_PWEAK" \
  | awk '{print $NF}')

echo "=== V3P M-E 并行探针已提交 ==="
echo "WSS-only · Job ${job_wss} · ${CFG_WSS}"
echo "PWeak    · Job ${job_pweak} · ${CFG_PWEAK}"
