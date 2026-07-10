#!/bin/bash
# 提交 WSS-min sweep：为清单里每个 config 提交一个 GPU 训练+评估作业。
# GPU 分区(master, 4×4090) 会自动排队，4 个并行。作业号写入 logs/submitted_<ts>.txt。
#
# 用法:
#   bash training_wss_min/cluster/submit_baseline_sweep.sh            # 提交全部
#   bash training_wss_min/cluster/submit_baseline_sweep.sh a.json b.json  # 只提交指定 config
#   WSSMIN_MANIFEST=training_wss_min/configs/sweep_round3_clean_v1.txt \
#     bash training_wss_min/cluster/submit_baseline_sweep.sh

set -euo pipefail
PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"
SBATCH=/public/slurm/bin/sbatch
MANIFEST="${WSSMIN_MANIFEST:-training_wss_min/configs/sweep_baseline_v1.txt}"
LOGDIR="training_wss_min/cluster/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
REC="${LOGDIR}/submitted_${STAMP}.txt"

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    mapfile -t CONFIGS < "$MANIFEST"
fi

echo "manifest=${MANIFEST}"
echo "提交 ${#CONFIGS[@]} 个作业，记录 -> ${REC}"
for cfg in "${CONFIGS[@]}"; do
    [ -z "$cfg" ] && continue
    name=$(basename "$cfg" .json)
    jid=$($SBATCH --parsable --job-name="wssmin_${name}" \
          training_wss_min/cluster/run_train.slurm "$cfg")
    echo "${jid}  ${name}  ${cfg}" | tee -a "$REC"
done

echo "---- 当前队列 ----"
/public/slurm/bin/squeue -u "$USER" -o "%.10i %.22j %.8T %.10M %.6D %R"
