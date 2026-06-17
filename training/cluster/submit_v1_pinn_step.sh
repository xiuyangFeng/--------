#!/usr/bin/env bash
# V1 PINN 阶梯 · 单档顺序提交（避免多作业同节点 OOM）
#
# 用法（仓库根）:
#   bash training/cluster/submit_v1_pinn_step.sh cont
#   bash training/cluster/submit_v1_pinn_step.sh contnoslip
#   bash training/cluster/submit_v1_pinn_step.sh full
#   bash training/cluster/submit_v1_pinn_step.sh all   # 仍顺序提交三档（非并行）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs training/cluster/logs

STEP="${1:-cont}"

declare -A CFG=(
  [cont]="training/configs/field/generated/v1_pinn/A-PINN-01-cont_seed1.json"
  [contnoslip]="training/configs/field/generated/v1_pinn/A-PINN-01-contnoslip_seed1.json"
  [full]="training/configs/field/generated/v1_pinn/A-PINN-01_seed1.json"
)
declare -A JNAME=(
  [cont]="v1pinn_cont"
  [contnoslip]="v1pinn_contnoslip"
  [full]="v1pinn_full_s1"
)

submit_one() {
  local step="$1"
  local cfg="${CFG[$step]}"
  local jname="${JNAME[$step]}"
  if [[ ! -f "$cfg" ]]; then
    echo "错误: 配置不存在: $cfg" >&2
    exit 1
  fi
  local jid
  jid=$(sbatch --chdir="$ROOT" --job-name="${jname}" \
    training/cluster/run_train_field.slurm \
    "${cfg}" \
    | awk '{print $NF}')
  echo "${step} -> Job ${jid}  (${cfg})"
  echo "  tail -f logs/${jname}_${jid}.out"
}

echo "=== V1 PINN 阶梯 · 单档提交 ==="
echo "工作目录: $ROOT"
echo "显存策略: batch_size=1 · physics.max_physics_nodes=2048"
echo ""

case "$STEP" in
  cont|contnoslip|full)
    submit_one "$STEP"
    ;;
  all)
    for s in cont contnoslip full; do
      submit_one "$s"
    done
    ;;
  *)
    echo "未知 step: $STEP（可选: cont | contnoslip | full | all）" >&2
    exit 1
    ;;
esac
