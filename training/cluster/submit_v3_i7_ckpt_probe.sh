#!/usr/bin/env bash
# V3P 路径 I7 · post5463 checkpoint 特征衰减探针（CPU）
# 用法（仓库根）:
#   bash training/cluster/submit_v3_i7_ckpt_probe.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

jid=$(sbatch --chdir="$ROOT" --job-name=v3p_i7_probe \
  training/cluster/run_v3_i7_ckpt_probe.slurm | awk '{print $NF}')

echo "=== V3P I7 ckpt probe 已提交 ==="
echo "Job ${jid}"
echo "  tail -f logs/v3p_i7_probe_${jid}.out"
