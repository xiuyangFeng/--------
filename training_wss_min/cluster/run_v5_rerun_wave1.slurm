#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --job-name=v5rerun_w1
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
# Wave 1 of the V5 rerun: R1/R2/R3 trained in parallel on physical GPUs 0,1,3 (GPU 2 is occupied by a
# non-Slurm process and is skipped on purpose), then test34 evaluation (best+last) and the legacy-checkpoint
# re-evaluation on the 34-case overlap.  All four GPUs are requested so that CUDA_VISIBLE_DEVICES indices
# are the physical ones.
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CFG=$PROJECT/training_wss_min/configs/v5_rerun_20260906
RUNS=$PROJECT/training_wss_min/runs/v5_rerun_20260906/outputs
LEGACY_SPLIT=$PROJECT/training_wss_min/experiments/v5_rerun_20260906/split_legacy_test36_overlap34.json
USE_GPUS=(${V5_GPUS:-0 1 3})
cd "$PROJECT"
echo "host=$(hostname) date=$(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
for g in "${USE_GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1500 ]; then echo "GPU $g already has ${used} MiB in use; refusing to start" >&2; exit 2; fi
done
declare -A EXP=( [0]=r1_lsa2_h2_logradius_s1234 [1]=r2_lsa2_mse_s1234 [2]=r3_s3_geope_mse_s1234 )
pids=()
for i in 0 1 2; do
  exp=${EXP[$i]}; gpu=${USE_GPUS[$i]}
  echo "launch $exp on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$PY" -m training_wss_min.train --config "$CFG/$exp.json" \
      > "training_wss_min/cluster/logs/${exp}_train.out" 2> "training_wss_min/cluster/logs/${exp}_train.err" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "at least one training failed" >&2; exit 1; }
echo "training done $(date)"
# test34 evaluation of the new runs (best and last checkpoints), sequential on the first chosen GPU
for i in 0 1 2; do
  exp=${EXP[$i]}
  for ck in best last; do
    CUDA_VISIBLE_DEVICES=${USE_GPUS[0]} "$PY" -m training_wss_min.evaluate --run-dir "$RUNS/$exp" --partitions test --allow-test --checkpoint "$ck" --no-plots \
        > "training_wss_min/cluster/logs/${exp}_eval_${ck}.out" 2>&1
  done
done
echo "new-run evaluation done $(date)"
# legacy checkpoints re-evaluated on the 34-case overlap (read-only, data_wss_min, stl_landmarks_v4)
declare -A LEG=( [r1]=pointnetpp_lsa2_h2_logradius/outputs/lsa2_h2_logradius_s1234 [r2]=pointnetpp_regp10_transformer/outputs/localtf_sa2_s1234 \
                 [r3]=pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed [r3s7]=pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed_s7 \
                 [r3s2025]=pointnetpp_d2_k64_ilo_structure/outputs/s3_d2k64_pnxr_geope_mixed_s2025 )
for key in r1 r2 r3 r3s7 r3s2025; do
  rd=$PROJECT/training_wss_min/runs/${LEG[$key]}
  out=$PROJECT/training_wss_min/experiments/v5_rerun_20260906/legacy_overlap34/${key}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=${USE_GPUS[0]} "$PY" -m training_wss_min.evaluate --run-dir "$rd" --partitions test --allow-test --checkpoint best --no-plots \
      --split-path "$LEGACY_SPLIT" --output-dir "$out" > "training_wss_min/cluster/logs/legacy_overlap34_${key}.out" 2>&1 || echo "legacy eval $key failed" >&2
done
echo "all done $(date)"
