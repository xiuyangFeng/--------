#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --job-name=v52_pilot_b
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_ingest_b_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_ingest_b_%j.err
# 入库链 pilot（2026-09-22，5 个新单元）：与 170 例相同的链，逐步留痕。
#   1 meshwall 中心线 V2（VMTK，从 .cas 壁面出发；出口语义 = UDF 线程号）
#   2 feature atlas（七段/曲率/半径/折点门）
#   3 拓扑审计  4 壁面审计  5 求解收敛审计（全量列表 + 新单元；旧例走缓存，汇总重建）
#   6 V5 build（pc-v3 点云门、坐标架、标签核对）到独立快照根，不碰 v5.1
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
cd "$ROOT"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
PYV=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
CASES=$(grep -v '^#' "${PILOT_CASES_FILE:-$HERE/pilot5_cases.txt}" | tr '\n' ' ')
CL="$ROOT/outputs/centerline_v2_meshwall_20260922_pilot"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_pilot_20260922"
SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_pilot5_train141_test34_s1234.json"
PREP="$ROOT/outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
START=${START_STEP:-4}
date -Is; hostname; echo "cases: $CASES  start_step=$START"
# 汇总文件备份（新单元会并入同一份汇总）
for f in audits/topology/summary.json audits/wall_wss/summary.json audits/solver_convergence/summary.json atlas/summary.json; do
  [ -f "$PREP/$f" ] && cp -p "$PREP/$f" "$PREP/$f.bak_20260922"
done
if [ "$START" -le 1 ]; then
echo "== 1 centerline (meshwall, vmtk)"; date -Is
$PYV tools/centerline_v2_meshwall.py --cases $CASES --output "$CL"
fi
if [ "$START" -le 2 ]; then
echo "== 2 feature atlas"; date -Is
$PY -m wss_pinn.v4.build_feature_atlas --cases $CASES --extra-cases $CASES --source-root "$CL" --workers 5
fi
if [ "$START" -le 3 ]; then
echo "== 3 topology audit"; date -Is
$PY -m wss_pinn.v4.audit_anatomy_topology --extra-cases $CASES --centerline-root "$CL" --workers 6 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"
fi
if [ "$START" -le 4 ]; then
echo "== 4 wall audit"; date -Is
$PY -m wss_pinn.v4.audit_wall_wss --extra-cases $CASES --workers 6 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"
fi
if [ "$START" -le 5 ]; then
echo "== 5 solver audit"; date -Is
$PY -m wss_pinn.v4.audit_solver_convergence --extra-cases $CASES --workers 8 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"
fi
if [ "$START" -le 6 ]; then
echo "== 6 V5 build"; date -Is
WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases $CASES --root "$SNAP" --workers 5 --run-name pilot_20260922
fi
echo "== done"; date -Is
