#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --job-name=v6_matrix
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=24:00:00
# Generic matrix runner (V6 single-frame WSS and later matrices).
#   V6_QUEUE   : space separated config stems under $V6_CFG_DIR (required)
#   V6_CFG_DIR : config directory (default configs/v6_singleframe_20260909)
#   V6_RUNS    : run root (default runs/<basename of V6_CFG_DIR>)
#   V6_GPUS    : physical GPU ids (default: the ids Slurm exposed to this job)
#   V6_SLOTS_PER_GPU : concurrent trainings per card (default 2)
#   V6_ALLOW_SHARED  : 1 = share cards with non-Slurm processes (skip the empty-card pre-check)
#   V6_MIN_FREE_MIB  : refuse to start a card with less free memory than this (default 8000)
# Trains SLOTS x GPUS at a time, then evaluates best and last on test34.
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CFG=${V6_CFG_DIR:-$PROJECT/training_wss_min/configs/v6_singleframe_20260909}
RUNS=${V6_RUNS:-$PROJECT/training_wss_min/runs/$(basename "$CFG")}
LOGS=$PROJECT/training_wss_min/cluster/logs
if [ -n "${V6_GPUS:-}" ]; then
  USE_GPUS=($V6_GPUS); GPU_SOURCE="explicit"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra USE_GPUS <<< "$CUDA_VISIBLE_DEVICES"; GPU_SOURCE="slurm"
else
  USE_GPUS=(0 1 2 3); GPU_SOURCE="default"
fi
QUEUE=(${V6_QUEUE:?V6_QUEUE required})
SLOTS=${V6_SLOTS_PER_GPU:-2}
MIN_FREE=${V6_MIN_FREE_MIB:-8000}
mkdir -p "$LOGS"
cd "$PROJECT"
echo "host=$(hostname) date=$(date) cfg=$CFG runs=$RUNS"
echo "queue=${QUEUE[*]}"
echo "gpus=${USE_GPUS[*]} (source=$GPU_SOURCE, slots/gpu=$SLOTS, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
# Shared-GPU mode: other users hold these cards outside Slurm, so only check that enough
# memory is left rather than demanding an empty card.
if [ "${V6_ALLOW_SHARED:-0}" = "1" ]; then
  for g in "${USE_GPUS[@]}"; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
    if [ "$free" -lt "$MIN_FREE" ]; then
      echo "GPU $g has only ${free} MiB free (< ${MIN_FREE}); refusing to start" >&2; exit 2
    fi
    echo "GPU $g: ${free} MiB free -> sharing"
  done
else
  for g in "${USE_GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -gt 1500 ]; then echo "GPU $g already has ${used} MiB in use; refusing to start" >&2; exit 2; fi
  done
fi
EXPANDED=(); for ((i=0;i<SLOTS;i++)); do EXPANDED+=("${USE_GPUS[@]}"); done; USE_GPUS=("${EXPANDED[@]}")
declare -A PID_GPU=()
launch() {
  echo "launch $1 on GPU $2 $(date +%T)"
  CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 "$PY" -m training_wss_min.train --config "$CFG/$1.json" \
    > "$LOGS/$1_train.out" 2> "$LOGS/$1_train.err" & PID_GPU[$!]=$2
}
fail=0; idx=0
for g in "${USE_GPUS[@]}"; do [ $idx -lt ${#QUEUE[@]} ] || break; launch "${QUEUE[$idx]}" "$g"; idx=$((idx+1)); sleep 5; done
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
  CUDA_VISIBLE_DEVICES=${USE_GPUS[0]} "$PY" -m training_wss_min.evaluate --run-dir "$RUNS/$exp" \
    --partitions test --allow-test --checkpoint "$ck" --no-plots > "$LOGS/${exp}_eval_${ck}.out" 2>&1 || fail=1
done; done
[ "$fail" -eq 0 ] || { echo "at least one evaluation failed" >&2; exit 1; }
echo "all done $(date)"
