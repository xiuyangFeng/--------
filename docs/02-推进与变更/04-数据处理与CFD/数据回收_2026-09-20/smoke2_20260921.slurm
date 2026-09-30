#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH -w node06
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=rerun_smoke2
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke2_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke2_%j.err
#SBATCH --time=01:00:00
# 补冒烟：SU_KAI_LI（路径改绝对后 read-case + UDF 编译）、PENG_JI_MING（read-case → mesh/scale → mesh/check，看域范围是否到 ~0.1 m 量级）。完整输出不截断。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
R=/public/newhome/cy/Digital_twin/GNN/data_new
for d in $R/AAA/ruputer/SU_KAI_LI $R/AG/fast/PENG_JI_MING; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  cat smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh > smoke_full.log 2>&1
  echo "===== fluent exit code $? ====="
  grep -n -i "Error\|not found\|libudf\|Domain Extents\|coordinate: min\|Warning: An error\|scale" smoke_full.log | head -40
  mv -f smoke_full.log "smoke_full_$(date +%H%M%S).log"; rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260921 2>/dev/null
done
