#!/bin/bash
# PF6 / VF6 v5.2d retrain (01 block tracking §43): the node04 share (configs/<exp>/matrix_node04.json, 11 arms) on node04's
# 2x A100, one arm per GPU, OUTSIDE Slurm. Start it right after submitting the Slurm preflight: it waits until the preflight
# has written a passed runtime_preflight.json, re-verifies the preflight fingerprints / arm coverage / anchor (the same check
# the master queue does), then runs run_local_train_queue --only <node04 arms> with its own status file
# (experiments/<exp>/queue_status_node04.json; the post job waits for its finished_at). Code: frozen copy.
#   ssh node04 "cd /public/newhome/cy/Digital_twin/GNN && nohup bash training_wss_min/cluster/node04_pf6vf6_v52d_queue.sh \
#       > training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/logs_node04_driver.log 2>&1 < /dev/null &"
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=pf6vf6_v52d_retrain_20261002
FROZEN=/public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
CFG=$ROOT/training_wss_min/configs/$NAME
X=$ROOT/training_wss_min/experiments/$NAME
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
GPUS=${GPUS:-0,1}
WAIT_HOURS=${WAIT_HOURS:-6}
cd "$FROZEN"
echo "$(date -Is) $(hostname) uid=$(id -u) gpus=$GPUS (outside Slurm)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
until [ -f "$X/runtime_preflight.json" ] && grep -q '"passed": true' "$X/runtime_preflight.json"; do
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "$(date -Is) no passed preflight after ${WAIT_HOURS} h; giving up"; exit 1; fi
  sleep 60
done
echo "$(date -Is) preflight passed; verifying fingerprints"
$PY - <<EOF
from pathlib import Path
from training_wss_min.tools.run_wss_local_wave1_queue import verify_preflight
print(verify_preflight(Path("$CFG"), Path("$X")))
EOF
ARMS=$($PY -c "import json; print(','.join(a['id'] for a in json.load(open('$CFG/matrix_node04.json'))['arms']))")
echo "$(date -Is) node04 arms: $ARMS"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY -u -m training_wss_min.tools.run_local_train_queue --config-dir "$CFG" --gpus "$GPUS" --slots-per-gpu 1 \
  --min-free-mib 6000 --launch-interval-seconds 45 --only "$ARMS" --status-file queue_status_node04.json
echo "$(date -Is) node04 queue finished"
