#!/usr/bin/env bash
# V3P M-E · 根据 oracle 波次结果提交结构性方向 GPU 训练（仅 Go + 已有 config）
#
# 前置: submit_v3_me_oracle_wave.sh → 判读 JSON → 在 ORACLE_GPU_REGISTRY 填入 gpu_config
#
# 用法:
#   bash training/cluster/submit_v3_me_structural_gpu.sh
#   bash training/cluster/submit_v3_me_structural_gpu.sh --dry-run
#   bash training/cluster/submit_v3_me_structural_gpu.sh --force O1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

REGISTRY="$ROOT/training/configs/field/me_oracle/ORACLE_GPU_REGISTRY.json"
ORACLE_JSON=""
DRY_RUN=0
FORCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --oracle-json) ORACLE_JSON="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [ -z "$ORACLE_JSON" ]; then
  ORACLE_JSON="$(ls -t "$ROOT"/outputs/field/f0_decision/v3p_me_oracle_wave_*.json 2>/dev/null | head -1 || true)"
fi
if [ -z "$ORACLE_JSON" ] || [ ! -f "$ORACLE_JSON" ]; then
  echo "错误: 找不到 oracle JSON"
  exit 1
fi

PYTHON="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
SLURM="$ROOT/training/cluster/run_train_field.slurm"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
command -v "$SBATCH_BIN" >/dev/null 2>&1 || SBATCH_BIN="/public/slurm/bin/sbatch"

echo "=== V3P M-E 结构性 GPU 提交 ==="
echo "Oracle: $ORACLE_JSON"

PLAN_FILE="$(mktemp)"
"$PYTHON" - "$ORACLE_JSON" "$REGISTRY" "$FORCE" >"$PLAN_FILE" <<'PY'
import json, sys
from pathlib import Path

oracle = json.loads(Path(sys.argv[1]).read_text())
registry = json.loads(Path(sys.argv[2]).read_text())
force = sys.argv[3].strip().upper()
force_set = {force} if force else set()

for oid in registry.get("priority_after_oracle", []):
    meta = registry["oracles"].get(oid)
    if not meta:
        continue
    if meta.get("status") == "completed_weak_no_go":
        print(f"SKIP|{oid}|completed|")
        continue
    block = oracle.get("oracles", {}).get(oid, {})
    verdict = block.get("verdict", "missing")
    cfg = meta.get("gpu_config") or ""
    if oid not in force_set and verdict != "go":
        print(f"SKIP|{oid}|{verdict}|")
        continue
    if not cfg:
        print(f"PENDING|{oid}|{verdict}|{meta.get('exp_id_stub','')}")
        continue
    print(f"SUBMIT|{oid}|{verdict}|{cfg}|{meta.get('exp_id_stub','')}")
PY

submitted=0
pending=0

while IFS='|' read -r action oid verdict rest; do
  case "$action" in
    SUBMIT)
      cfg_rel="$rest"
      cfg_full="$ROOT/$cfg_rel"
      if [ ! -f "$cfg_full" ]; then
        echo "SKIP $oid · 缺少 config: $cfg_full"
        continue
      fi
      job_name="v3p_me_${oid,,}"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN · $oid · $cfg_rel"
      else
        jid=$("$SBATCH_BIN" -p GPU --gres=gpu:1 --job-name="$job_name" \
          --chdir="$ROOT" \
          --output="$ROOT/logs/${job_name}_%j.out" \
          --error="$ROOT/logs/${job_name}_%j.err" \
          "$SLURM" "$cfg_full" | awk '{print $NF}')
        echo "SUBMITTED $oid · Job $jid · $cfg_rel"
      fi
      submitted=$((submitted + 1))
      ;;
    PENDING)
      echo "PENDING $oid · Go 待建 config · stub: $rest"
      pending=$((pending + 1))
      ;;
    SKIP)
      echo "SKIP $oid · $verdict"
      ;;
  esac
done < "$PLAN_FILE"
rm -f "$PLAN_FILE"

echo "汇总: submitted=$submitted · pending_config=$pending"
