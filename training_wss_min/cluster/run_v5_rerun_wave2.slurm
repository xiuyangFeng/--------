#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --job-name=v5rerun_w2
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
# Wave 2: seed replicates of R1/R3 (7, 2025) + R4 (R1 + V5 point features). Five trainings scheduled over
# physical GPUs 0,1,3 (GPU 2 skipped), three at a time; then best/last test34 evaluation of each run.
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CFG=$PROJECT/training_wss_min/configs/v5_rerun_20260906
RUNS=$PROJECT/training_wss_min/runs/v5_rerun_20260906/outputs
USE_GPUS=(${V5_GPUS:-0 1 3})
QUEUE=(r1_lsa2_h2_logradius_s7 r3_s3_geope_mse_s7 r4_lsa2_h2_logradius_v5feat_s1234 r1_lsa2_h2_logradius_s2025 r3_s3_geope_mse_s2025)
cd "$PROJECT"
echo "host=$(hostname) date=$(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
for g in "${USE_GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1500 ]; then echo "GPU $g already has ${used} MiB in use; refusing to start" >&2; exit 2; fi
done
declare -A PID_GPU=()
launch() {  # $1=exp $2=gpu
  echo "launch $1 on GPU $2 $(date +%T)"
  CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 "$PY" -m training_wss_min.train --config "$CFG/$1.json" \
      > "training_wss_min/cluster/logs/$1_train.out" 2> "training_wss_min/cluster/logs/$1_train.err" &
  PID_GPU[$!]=$2
}
fail=0
idx=0
for g in "${USE_GPUS[@]}"; do
  [ $idx -lt ${#QUEUE[@]} ] || break
  launch "${QUEUE[$idx]}" "$g"; idx=$((idx+1))
done
while [ ${#PID_GPU[@]} -gt 0 ]; do
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || fail=1
      freed=${PID_GPU[$pid]}; unset "PID_GPU[$pid]"
      if [ $idx -lt ${#QUEUE[@]} ]; then launch "${QUEUE[$idx]}" "$freed"; idx=$((idx+1)); fi
    fi
  done
  sleep 15
done
[ "$fail" -eq 0 ] || { echo "at least one training failed" >&2; exit 1; }
echo "training done $(date)"
for exp in "${QUEUE[@]}"; do
  for ck in best last; do
    CUDA_VISIBLE_DEVICES=${USE_GPUS[0]} "$PY" -m training_wss_min.evaluate --run-dir "$RUNS/$exp" --partitions test --allow-test --checkpoint "$ck" --no-plots \
        > "training_wss_min/cluster/logs/${exp}_eval_${ck}.out" 2>&1
  done
done
echo "all done $(date)"
