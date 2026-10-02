#!/bin/bash
# throughput benchmark: <prefix> <gpu_solo> <gpu_pair>; PF6 cv5 fold0 recipe on 60 train units, 8 epochs (temporary runs)
set -u
cd /public/newhome/cy/Digital_twin/GNN
B=training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/bench
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
hostname; date -Is; nvidia-smi --query-gpu=index,name,memory.used --format=csv
CUDA_VISIBLE_DEVICES=$2 $PY -u -m training_wss_min.train --config $B/${1}_solo.json > $B/${1}_solo.log 2>&1 &
CUDA_VISIBLE_DEVICES=$3 $PY -u -m training_wss_min.train --config $B/${1}_pairA.json > $B/${1}_pairA.log 2>&1 &
CUDA_VISIBLE_DEVICES=$3 $PY -u -m training_wss_min.train --config $B/${1}_pairB.json > $B/${1}_pairB.log 2>&1 &
wait; date -Is
