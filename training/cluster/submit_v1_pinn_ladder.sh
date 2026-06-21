#!/usr/bin/env bash
# V1 PINN 阶梯消融 · SLURM 提交（seed1 三档并行）
#
# 阶梯顺序（逻辑上 cont → cont+noslip → full；本脚本一次提交 3 个独立作业便于 squeue 监控）:
#   1. A-PINN-01-cont        仅连续性
#   2. A-PINN-01-contnoslip  连续性 + 物理 no-slip
#   3. A-PINN-01             全量 PINN（+ 定常动量）
#
# 用法（仓库根）:
#   bash training/cluster/submit_v1_pinn_ladder.sh
#   bash training/cluster/submit_v1_pinn_ladder.sh --dry-run
#
# 监控:
#   squeue -u $USER
#   tail -f logs/v1pinn_cont_<JOB_ID>.out        # 仓库根 logs/（%x_%j 命名）
#   tail -f logs/v1pinn_contnoslip_<JOB_ID>.out
#   tail -f logs/v1pinn_full_s1_<JOB_ID>.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p training/cluster/logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

submit_one() {
  local job_name="$1"
  local cfg="$2"
  if [[ ! -f "$cfg" ]]; then
    echo "错误: 配置文件不存在: $cfg" >&2
    exit 1
  fi
  if $DRY_RUN; then
    echo "[dry-run] sbatch --job-name=${job_name} training/cluster/run_train_field.slurm ${cfg}"
    return 0
  fi
  local jid
  jid=$(sbatch --job-name="${job_name}" \
    training/cluster/run_train_field.slurm \
    "${cfg}" \
    | awk '{print $NF}')
  echo "${job_name} -> Job ${jid}  (${cfg})"
}

echo "=== V1 PINN 阶梯消融（SLURM · seed1 × 3）==="
echo "工作目录: $ROOT"
echo "环境: ${TRAINING_ENV:-GNN}"
echo ""

submit_one "v1pinn_cont" \
  "training/configs/field/generated/v1_pinn/A-PINN-01-cont_seed1.json"
submit_one "v1pinn_contnoslip" \
  "training/configs/field/generated/v1_pinn/A-PINN-01-contnoslip_seed1.json"
submit_one "v1pinn_full_s1" \
  "training/configs/field/generated/v1_pinn/A-PINN-01_seed1.json"

echo ""
if $DRY_RUN; then
  echo "dry-run 完成，未实际提交。"
else
  echo "已提交 3 个作业。监控: squeue -u \$USER"
  echo "日志目录: logs/v1pinn_<name>_<JOB_ID>.out（仓库根）"
fi
