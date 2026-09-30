#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --job-name=voltime_nmae
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 体场时间臂 ckpt_best 周期重评：写入逐帧 pooled nmae_range。
# 只评 PT0/PTB8/VT0/VTB4 × 3 折，结果写独立目录，不覆盖原 metrics.json。
set -u
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$MAIN"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUNS=$MAIN/training_wss_min/runs/volume_time_20260919
OUT=$MAIN/training_wss_min/experiments/volume_time_20260919/cycle_nmae_20260920
LOG=$OUT/logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPUS=(0 1 2 3)
fi
echo "worker GPUs: ${GPUS[*]}"

ARMS=(
  PT0_f0_s1234 PT0_f1_s1234 PT0_f2_s1234
  PTB8_f0_s1234 PTB8_f1_s1234 PTB8_f2_s1234
  VT0_f0_s1234 VT0_f1_s1234 VT0_f2_s1234
  VTB4_f0_s1234 VTB4_f1_s1234 VTB4_f2_s1234
)

eval_one() {
  local arm=$1 gpu=$2
  echo "[$(date -Is)] start $arm gpu=$gpu"
  mkdir -p "$OUT/$arm"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m training_wss_min.evaluate \
    --run-dir "$RUNS/$arm" \
    --partitions test --allow-test --checkpoint best --no-plots \
    --output-dir "$OUT/$arm" \
    > "$LOG/${arm}.out" 2> "$LOG/${arm}.err"
  local rc=$?
  echo "[$(date -Is)] done $arm gpu=$gpu rc=$rc"
  return $rc
}

fail=0
gpu_pid=()
gpu_arm=()
n_gpu=${#GPUS[@]}
for ((g=0; g<n_gpu; g++)); do
  gpu_pid[$g]=""
  gpu_arm[$g]=""
done

reap_gpu() {
  local g=$1
  local pid=${gpu_pid[$g]}
  [ -z "$pid" ] && return 0
  if wait "$pid"; then
    echo "[$(date -Is)] reaped ${gpu_arm[$g]} gpu=${GPUS[$g]} rc=0"
  else
    echo "[$(date -Is)] FAILED ${gpu_arm[$g]} gpu=${GPUS[$g]}"
    fail=1
  fi
  gpu_pid[$g]=""
  gpu_arm[$g]=""
}

for arm in "${ARMS[@]}"; do
  while true; do
    for ((g=0; g<n_gpu; g++)); do
      pid=${gpu_pid[$g]}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        if [ -n "$pid" ]; then
          reap_gpu "$g"
        fi
        eval_one "$arm" "${GPUS[$g]}" &
        gpu_pid[$g]=$!
        gpu_arm[$g]=$arm
        break 2
      fi
    done
    sleep 5
  done
done

for ((g=0; g<n_gpu; g++)); do
  [ -n "${gpu_pid[$g]}" ] && reap_gpu "$g"
done
date -Is
nvidia-smi --query-gpu=index,memory.used --format=csv
if [ "$fail" -ne 0 ]; then
  echo "one or more cycle NMAE evals failed"
  exit 1
fi
echo "all 12 cycle NMAE evals finished"
