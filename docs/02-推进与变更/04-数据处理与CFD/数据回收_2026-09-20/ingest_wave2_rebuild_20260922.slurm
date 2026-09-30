#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --job-name=v52_rebuild
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/ingest_wave2_rebuild_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/ingest_wave2_rebuild_%j.err
# 2026-09-22：15528 结束后，用改过壳层门的 build 对第 2 波未过门的单元重建（--resume 跳过已过门的），快照根不变。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"
SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json"
CASES=$(tr '\n' ' ' < "$HERE/ingest_wave2_cl_pass.txt")
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname
WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases $CASES --root "$SNAP" --workers 6 --resume --run-name new95_wave2_rebuild_shell_20260922
echo "== done"; date -Is
