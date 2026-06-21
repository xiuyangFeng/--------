#!/usr/bin/env bash
# V3P 路径 I · Q0 诊断提交（0 GPU · node03 CPU）
#
# 用法（仓库根）:
#   bash training/cluster/submit_v3_q0_diagnostics.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

jid=$(sbatch --chdir="$ROOT" --job-name=v3p_q0_diag \
  training/cluster/run_v3_q0_diagnostics.slurm | awk '{print $NF}')

echo "=== V3P Q0 诊断已提交 ==="
echo "Job ${jid} · I-oracle bundle + I7 ckpt probe"
echo "  tail -f logs/v3p_q0_diag_${jid}.out"
