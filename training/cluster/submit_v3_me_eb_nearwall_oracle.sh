#!/usr/bin/env bash
# V3P M-E E-B 近壁算子 oracle 提交（CPU · node03）
#
# 用法: bash training/cluster/submit_v3_me_eb_nearwall_oracle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/field/f0_decision

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  else
    echo "本地运行:"
    echo "  /public/newhome/cy/.conda/envs/GNN/bin/python -m training.scripts.run_v3_me_eb_nearwall_oracle $*"
    exit 1
  fi
fi

EXTRA_ARGS=("$@")
job_id=$("$SBATCH_BIN" --chdir="$ROOT" --job-name=v3p_me_eb \
  training/cluster/run_v3_me_eb_nearwall_oracle.slurm \
  "${EXTRA_ARGS[@]}" 2>&1 | awk '{print $NF}')

echo "=== E-B near-wall oracle 已提交 ==="
echo "Job ${job_id} · tail -f logs/v3p_me_eb_${job_id}.out"
echo "产物: outputs/field/f0_decision/v3p_me_eb_nearwall_oracle_<date>.json"
