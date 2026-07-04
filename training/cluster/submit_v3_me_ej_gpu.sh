#!/usr/bin/env bash
# V3P M-E E-J · oracle weak_go/go 后提交 PGradBridge GPU probe
#
# 用法:
#   bash training/cluster/submit_v3_me_ej_gpu.sh
#   bash training/cluster/submit_v3_me_ej_gpu.sh --dry-run
#   bash training/cluster/submit_v3_me_ej_gpu.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

ORACLE_JSON=""
DRY_RUN=0
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --oracle-json) ORACLE_JSON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [ -z "$ORACLE_JSON" ]; then
  ORACLE_JSON="$(ls -t "$ROOT"/outputs/field/f0_decision/v3p_me_ej_oracle_*.json 2>/dev/null | head -1 || true)"
fi
if [ ! -f "$ORACLE_JSON" ]; then
  echo "错误: 找不到 E-J oracle JSON，请先运行 run_v3_me_ej_oracle"
  exit 1
fi

CFG="$ROOT/training/configs/field/generated/v3_pointcloud/V3P-M-E-EJ-PGradBridge-post5463_seed1.json"
SLURM="$ROOT/training/cluster/run_train_field.slurm"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
command -v "$SBATCH_BIN" >/dev/null 2>&1 || SBATCH_BIN="/public/slurm/bin/sbatch"

VERDICT="$(python3 -c "import json; d=json.load(open('$ORACLE_JSON')); print(d.get('verdict',''))")"
ALLOW=0
if [ "$FORCE" -eq 1 ]; then ALLOW=1; fi
if [ "$VERDICT" = "go" ] || [ "$VERDICT" = "weak_go" ]; then ALLOW=1; fi

echo "=== V3P M-E E-J GPU probe ==="
echo "Oracle: $ORACLE_JSON · verdict=$VERDICT"

if [ "$ALLOW" -ne 1 ]; then
  echo "SKIP: verdict 非 go/weak_go（需 --force 强制）"
  exit 0
fi

if [ ! -f "$CFG" ]; then
  echo "错误: 缺少 config $CFG"
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY-RUN · would submit $CFG"
  exit 0
fi

jid=$("$SBATCH_BIN" -p GPU --gres=gpu:1 --job-name=v3p_me_ej \
  --chdir="$ROOT" \
  --output="$ROOT/logs/v3p_me_ej_%j.out" \
  --error="$ROOT/logs/v3p_me_ej_%j.err" \
  "$SLURM" "$CFG" | awk '{print $NF}')

echo "SUBMITTED E-J PGradBridge · Job $jid"
echo "tail -f logs/v3p_me_ej_${jid}.out"
