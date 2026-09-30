#!/bin/bash
# 2026-09-27: run a training matrix on node04's two A100 (NOT under Slurm). The preflight/queue tools require Slurm-style
# environment variables; we set them by hand and record the manual tag in the evidence (slurm_job_id = node04_manual_<stamp>).
# usage: bash node04_run_matrix.sh <NAME> <control-arm> <smoke-arms> <slots-per-gpu> [gpus=0,1]
set -euo pipefail
NAME=$1; CTRL=$2; SMOKE=$3; SLOTS=$4; GPUS=${5:-0,1}
ROOT=/public/newhome/cy/Digital_twin/GNN
FROZEN=${FROZEN:-/public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926}
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
STAMP=$(date +%Y%m%dT%H%M%S)
# the queue tool insists on a NUMERIC SLURM_JOB_ID: node04 manual runs use a 9-prefixed pseudo id 9MMDDHHMM (real Slurm ids are 5 digits)
export SLURM_JOB_ID="9$(date +%m%d%H%M)" SLURM_GPUS_ON_NODE=$(echo "$GPUS" | tr ',' '\n' | wc -l) CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$FROZEN"
LOGDIR="$ROOT/training_wss_min/cluster/logs"; mkdir -p "$LOGDIR"
echo "$(date -Is) $(hostname) NAME=$NAME tag=$SLURM_JOB_ID gpus=$GPUS slots=$SLOTS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
if [ "${SKIP_PREFLIGHT:-0}" = "1" ] && grep -q '"passed": true' "$ROOT/training_wss_min/experiments/$NAME/runtime_preflight.json" 2>/dev/null; then echo "== preflight skipped (existing passed evidence)"; else
echo "== preflight"; date -Is
$PY -u -m training_wss_min.tools.preflight_wss_local_wave1 --config-dir "$ROOT/training_wss_min/configs/$NAME" --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
   --control-arm "$CTRL" --smoke-arms "$SMOKE" --anchor-run "$ROOT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234" > "$LOGDIR/node04_${NAME}_preflight_${STAMP}.log" 2>&1
grep -q "PREFLIGHT PASSED" "$LOGDIR/node04_${NAME}_preflight_${STAMP}.log" || { echo "preflight failed, see $LOGDIR/node04_${NAME}_preflight_${STAMP}.log"; exit 1; }
fi
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then echo "== preflight only, no queue"; date -Is; exit 0; fi
echo "== queue"; date -Is
export OMP_NUM_THREADS=2
$PY -u -m training_wss_min.tools.run_wss_local_wave1_queue --config-dir "$ROOT/training_wss_min/configs/$NAME" --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
   --slots-per-gpu "$SLOTS" --min-free-mib 7000 --launch-interval-seconds 45 > "$LOGDIR/node04_${NAME}_queue_${STAMP}.log" 2>&1
echo "== done"; date -Is
