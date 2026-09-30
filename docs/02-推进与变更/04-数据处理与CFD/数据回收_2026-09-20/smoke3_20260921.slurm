#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH -w node03
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=rerun_smoke3
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke3_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke3_%j.err
#SBATCH --time=03:00:00
# 补冒烟：SU_KAI_LI（路径改绝对后 read-case + UDF 编译）、PENG_JI_MING（read-case → mesh/scale → mesh/check，看域范围是否到 ~0.1 m 量级）。完整输出不截断。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
R=/public/newhome/cy/Digital_twin/GNN/data_new
for d in $R/ILO/WANG_JIN_MING-0/after $R/ILO/ZHANG_JIN_CHUN-1/after $R/ILO/ZHANG_YONG_SHENG-0/after $R/ILO/ZHAO_JIAN_PING-0/after $R/ILO/ZHAO_JIAN_PING-0/before $R/ILO/SHEN_CHUN_WANG-0/before $R/ILO/ZHANG_YAN_SHAN-0/after $R/ILO/YANG_QING_REN-1/before $R/ILO/LU_FU_SHAN-0/after $R/ILO/SUN_XU_XIA-1/after $R/ILO/SUN_YU_SHENG-0/after $R/ILO/GUO_AI_JUN-0/after $R/ILO/GUO_AI_JUN-0/before $R/ILO/LIU_BAO_JUN-0/after $R/ILO/XUE_YOU_TANG-0/before ; do
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
