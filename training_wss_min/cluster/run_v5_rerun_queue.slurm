#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --job-name=v5rerun_q
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
# Generic queued runner: V5_QUEUE="exp1 exp2 ..." (config names under configs/v5_rerun_20260906).
# GPUs: V5_GPUS="0 1 3" pins physical ids (use with --gres=gpu:4); when unset, the job uses the GPUs Slurm allocated
# (submit with e.g. `sbatch --gres=gpu:2 --mem=56G`). Trains N at a time, then evaluates best/last on test34.
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CFG=$PROJECT/training_wss_min/configs/v5_rerun_20260906
RUNS=$PROJECT/training_wss_min/runs/v5_rerun_20260906/outputs
# GPU 选择：显式 V5_GPUS（物理编号，需 --gres=gpu:4 独占整机时使用）优先；否则用 Slurm 分配给本作业的
# CUDA_VISIBLE_DEVICES（--gres=gpu:N 时由 Slurm 保证独占，不再做显存预检）。
if [ -n "${V5_GPUS:-}" ]; then
  USE_GPUS=($V5_GPUS); GPU_SOURCE="explicit"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra USE_GPUS <<< "$CUDA_VISIBLE_DEVICES"; GPU_SOURCE="slurm"
else
  USE_GPUS=(0 1 3); GPU_SOURCE="default"
fi
QUEUE=(${V5_QUEUE:?V5_QUEUE required})
# 每张卡并发几个训练（默认 1）。共享显存时用 2 可以在别人占满 SM 的情况下多拿一些时间片。
SLOTS=${V5_SLOTS_PER_GPU:-1}
if [ "$SLOTS" -gt 1 ]; then
  EXPANDED=(); for ((i=0;i<SLOTS;i++)); do EXPANDED+=("${USE_GPUS[@]}"); done; USE_GPUS=("${EXPANDED[@]}")
fi
cd "$PROJECT"
echo "host=$(hostname) date=$(date) queue=${QUEUE[*]} gpus=${USE_GPUS[*]} (source=$GPU_SOURCE, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
# 显存预检（用户要求只用空卡）。Slurm 只保证 Slurm 作业之间独占，节点上仍可能有非 Slurm 进程占卡。
# 注意索引口径：在 Slurm 模式下 cgroup 只暴露本作业分到的卡且重新编号 0..N-1，nvidia-smi 在作业内看到的
# 就是这批卡，因此直接检查"作业内可见的全部 GPU"；显式 V5_GPUS 时才按物理编号逐个检查。
if [ "${V5_ALLOW_SHARED:-0}" = "1" ]; then
  echo "V5_ALLOW_SHARED=1: skipping the empty-GPU pre-check; this job will share the cards with other processes"
elif [ "$GPU_SOURCE" = "slurm" ]; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>1500 {print $1":"$2"MiB"}')
  if [ -n "$busy" ]; then echo "allocated GPU(s) already in use by another process: $busy; refusing to start" >&2; exit 2; fi
else
  for g in "${USE_GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -gt 1500 ]; then echo "GPU $g already has ${used} MiB in use; refusing to start" >&2; exit 2; fi
  done
fi
declare -A PID_GPU=()
launch() { echo "launch $1 on GPU $2 $(date +%T)"; CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 "$PY" -m training_wss_min.train --config "$CFG/$1.json" > "training_wss_min/cluster/logs/$1_train.out" 2> "training_wss_min/cluster/logs/$1_train.err" & PID_GPU[$!]=$2; }
fail=0; idx=0
for g in "${USE_GPUS[@]}"; do [ $idx -lt ${#QUEUE[@]} ] || break; launch "${QUEUE[$idx]}" "$g"; idx=$((idx+1)); done
while [ ${#PID_GPU[@]} -gt 0 ]; do
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || fail=1; freed=${PID_GPU[$pid]}; unset "PID_GPU[$pid]"
      if [ $idx -lt ${#QUEUE[@]} ]; then launch "${QUEUE[$idx]}" "$freed"; idx=$((idx+1)); fi
    fi
  done
  sleep 15
done
[ "$fail" -eq 0 ] || { echo "at least one training failed" >&2; exit 1; }
echo "training done $(date)"
for exp in "${QUEUE[@]}"; do for ck in best last; do
  CUDA_VISIBLE_DEVICES=${USE_GPUS[0]} "$PY" -m training_wss_min.evaluate --run-dir "$RUNS/$exp" --partitions test --allow-test --checkpoint "$ck" --no-plots > "training_wss_min/cluster/logs/${exp}_eval_${ck}.out" 2>&1
done; done
echo "all done $(date)"
