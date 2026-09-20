#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=84G
#SBATCH --job-name=eval_only
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=8:00:00
# Parallel evaluation of already-trained runs (checkpoints are on disk; no training here).
#   V5_EVAL_RUNS="run1 run2 ..."   run directory names under runs/<GROUP>/outputs
#   V5_GROUP=v5_rerun_20260906     run group (default)
#   V5_CKPTS="best last"           checkpoints to evaluate
#   V5_SLOTS_PER_GPU=2             concurrent evaluations per GPU
#   V5_ALLOW_SHARED=1              skip the empty-GPU pre-check (share the cards with other processes)
set -u
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
GROUP=${V5_GROUP:-v5_rerun_20260906}
RUNS_DIR=$PROJECT/training_wss_min/runs/$GROUP/outputs
CKPTS=(${V5_CKPTS:-best last})
RUNS=(${V5_EVAL_RUNS:?V5_EVAL_RUNS required})
if [ -n "${V5_GPUS:-}" ]; then USE_GPUS=($V5_GPUS)
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=',' read -ra USE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
else USE_GPUS=(0); fi
SLOTS=${V5_SLOTS_PER_GPU:-1}
if [ "$SLOTS" -gt 1 ]; then EXP=(); for ((i=0;i<SLOTS;i++)); do EXP+=("${USE_GPUS[@]}"); done; USE_GPUS=("${EXP[@]}"); fi
cd "$PROJECT"
echo "host=$(hostname) date=$(date) runs=${#RUNS[@]} ckpts=${CKPTS[*]} slots=${USE_GPUS[*]}"
if [ "${V5_ALLOW_SHARED:-0}" != "1" ]; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2>1500 {print $1}')
  if [ -n "$busy" ]; then echo "GPU(s) $busy busy; refusing (set V5_ALLOW_SHARED=1 to share)" >&2; exit 2; fi
fi
JOBS=(); for r in "${RUNS[@]}"; do for c in "${CKPTS[@]}"; do JOBS+=("$r|$c"); done; done
echo "total evaluations: ${#JOBS[@]}"
declare -A PID_GPU=(); fail=0; idx=0
launch() {
  local run="${1%%|*}" ck="${1##*|}" gpu="$2"
  echo "eval $run/$ck on GPU $gpu $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUNS_DIR/$run" --partitions test --allow-test --checkpoint "$ck" --no-plots \
    > "training_wss_min/cluster/logs/eval_${run}_${ck}.out" 2>&1 &
  PID_GPU[$!]=$gpu
}
for g in "${USE_GPUS[@]}"; do [ $idx -lt ${#JOBS[@]} ] || break; launch "${JOBS[$idx]}" "$g"; idx=$((idx+1)); done
while [ ${#PID_GPU[@]} -gt 0 ]; do
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || fail=$((fail+1)); freed=${PID_GPU[$pid]}; unset "PID_GPU[$pid]"
      if [ $idx -lt ${#JOBS[@]} ]; then launch "${JOBS[$idx]}" "$freed"; idx=$((idx+1)); fi
    fi
  done
  sleep 10
done
echo "done $(date); failures=$fail"
[ "$fail" -eq 0 ]
