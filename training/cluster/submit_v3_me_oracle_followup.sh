#!/usr/bin/env bash
# V3P M-E oracle 补跑一键提交：① I6-diag val GPU 预测 ② O9/O3/O11 CPU followup
#
# 用法（仓库根）:
#   bash training/cluster/submit_v3_me_oracle_followup.sh
#   bash training/cluster/submit_v3_me_oracle_followup.sh --local-o9-o3   # 仅 CPU（跳过 O11）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/field/f0_decision

LOCAL_O9_O3_ONLY=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --local-o9-o3) LOCAL_O9_O3_ONLY=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc"
fi

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
  if [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
  fi
fi

PYTHON="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
RUN_DIR="outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
VAL_MANIFEST="$ROOT/$RUN_DIR/predictions_val_best_wss/manifest.json"

echo "=== V3P M-E oracle followup 提交 ==="

if [ "$LOCAL_O9_O3_ONLY" -eq 1 ] || [ ! -x "$(command -v "$SBATCH_BIN" 2>/dev/null || true)" ] && [ ! -x /public/slurm/bin/sbatch ]; then
  echo "本地/仅 O9+O3 模式"
  "$PYTHON" -m training.scripts.run_v3_me_oracle_followup \
    --skip-o11 "${EXTRA[@]}"
  exit 0
fi

if [ -f "$VAL_MANIFEST" ]; then
  echo "val 预测已存在: $VAL_MANIFEST · 跳过 GPU，直接提交 followup CPU"
  val_job=""
else
  val_job=$("$SBATCH_BIN" --chdir="$ROOT" --job-name=v3p_i6d_valpred \
    training/cluster/run_v3_me_i6diag_val_predict.slurm | awk '{print $NF}')
  echo "val 预测 GPU · Job ${val_job}"
fi

if [ -n "${val_job:-}" ]; then
  follow_job=$("$SBATCH_BIN" --chdir="$ROOT" --job-name=v3p_me_followup \
    --dependency=afterok:"$val_job" \
    training/cluster/run_v3_me_oracle_followup.slurm \
    "${EXTRA[@]}" | awk '{print $NF}')
else
  follow_job=$("$SBATCH_BIN" --chdir="$ROOT" --job-name=v3p_me_followup \
    training/cluster/run_v3_me_oracle_followup.slurm \
    "${EXTRA[@]}" | awk '{print $NF}')
fi

echo ""
echo "followup CPU · Job ${follow_job}"
echo "  tail -f logs/v3p_me_followup_${follow_job}.out"
echo "产物: outputs/field/f0_decision/v3p_me_oracle_followup_<date>.json"
