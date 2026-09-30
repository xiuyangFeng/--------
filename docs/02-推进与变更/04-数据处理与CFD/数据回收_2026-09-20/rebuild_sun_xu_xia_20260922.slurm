#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=v52_rebuild_sxx
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/rebuild_sxx_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/rebuild_sxx_%j.err
# SUN_XU_XIA-1/after：用户 2026-09-22 豁免 pca_normals_ok（翻转 0.517% vs 0.5%，全在并排髂支 1.3 mm 处）→ contract.CASE_GATE_WAIVERS，重建报告使 pass=True（bundle 内容不变）。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"; SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname
WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases ILO/SUN_XU_XIA-1/after --root "$SNAP" --workers 1 --run-name sun_xu_xia_waiver_20260922
$PY -c "import json; r=json.load(open('$SNAP/cases/ILO__SUN_XU_XIA-1__after/report.json')); print('gates', r['gates'])"
date -Is
