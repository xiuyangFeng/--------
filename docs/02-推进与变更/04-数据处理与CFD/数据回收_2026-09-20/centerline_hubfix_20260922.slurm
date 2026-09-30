#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=cl_hubfix
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/centerline_hubfix_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/centerline_hubfix_%j.err
# 2026-09-22：解剖区按拓扑（hub）解析后，为 NIE_QUAN_ZHONG（blood 是入口延伸段，解剖区 = blood2）与 WANG_CAI-0/before（fluid-NNNN 自动命名）补算 meshwall 中心线，
# 输出到独立根 centerline_v2_meshwall_20260922_wave3（第 3 波：这 2 例 + 第三批 13 例 CFD 的中心线也写这里）。
set -uo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"
PYV=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
export OMP_NUM_THREADS=4 PYTHONPATH="$ROOT"
date -Is; hostname
$PYV tools/centerline_v2_meshwall.py --cases AG/slow/NIE_QUAN_ZHONG ILO/WANG_CAI-0/before --output "$ROOT/outputs/centerline_v2_meshwall_20260922_wave3"; echo "exit $?"
date -Is
