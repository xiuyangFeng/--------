#!/usr/bin/env bash
# V3P 路径 I6-a · 两阶段冻结 wss_head 探针（seed1 · M-E O6）
# 用法（仓库根）:
#   bash training/cluster/submit_v3_i6a_two_stage.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

CFG="$ROOT/training/configs/field/generated/v3_pointcloud/V3P-I6-a-AsymW-a-post5463_seed1.json"
CKPT="$ROOT/outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001/checkpoint_epoch_10.pt"
SLURM="$ROOT/training/cluster/run_train_field.slurm"

if [ ! -f "$CFG" ]; then
  echo "错误: 配置文件不存在: $CFG"
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "错误: I6-diag warm-start ckpt 不存在: $CKPT"
  exit 1
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

job_id=$("$SBATCH_BIN" -p GPU --gres=gpu:1 --job-name="v3p_i6a_s1" \
  --chdir="$ROOT" \
  --output="$ROOT/logs/v3p_i6a_s1_%j.out" \
  --error="$ROOT/logs/v3p_i6a_s1_%j.err" \
  "$SLURM" \
  "$CFG" \
  | awk '{print $NF}')

echo "已提交 V3P-I6-a seed1 · Job ${job_id}"
echo "配置: ${CFG}"
echo "warm-start: ${CKPT}"
