#!/usr/bin/env bash
# V3P M-E · O1–O12 零重训 oracle 波次提交（CPU · node03）
#
# 用法（仓库根）:
#   bash training/cluster/submit_v3_me_oracle_wave.sh
#   bash training/cluster/submit_v3_me_oracle_wave.sh --only O1,O3,O10
#   bash training/cluster/submit_v3_me_oracle_wave.sh --skip O6

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/field/f0_decision

if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  else
    echo "错误: 找不到 sbatch；可本地跑:"
    echo "  /public/newhome/cy/.conda/envs/GNN/bin/python -m training.scripts.run_v3_me_oracle_wave $*"
    exit 1
  fi
fi

EXTRA_ARGS=("$@")
job_id=$("$SBATCH_BIN" --chdir="$ROOT" --job-name=v3p_me_oracle \
  training/cluster/run_v3_me_oracle_wave.slurm \
  "${EXTRA_ARGS[@]}" 2>&1 | awk '{print $NF}')

echo "=== V3P M-E oracle 波次已提交 ==="
echo "Job ${job_id} · Slurm run_v3_me_oracle_wave.slurm"
echo "  tail -f logs/v3p_me_oracle_${job_id}.out"
echo ""
echo "产物: outputs/field/f0_decision/v3p_me_oracle_wave_<date>.json"
echo "判读后 GPU 立项: bash training/cluster/submit_v3_me_structural_gpu.sh"
