#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH -w node06
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=rerun_smoke
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke_%j.err
#SBATCH --time=02:00:00
# 冒烟：逐例只 read-case（触发 libudf 从修正后的源码重编译）→ /mesh/check（打印域范围，验证 PENG_JI_MING 缩放）→ 退出，不迭代、不导出。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
R=/public/newhome/cy/Digital_twin/GNN/data_new
for d in $R/AAA/ruputer/LV_GUO_YOU $R/AG/slow/LIU_XI_QUAN $R/AG/slow/WEI_JUN_WEN $R/AG/slow/TE_JIN_WANG $R/ILO/WANG_CAI-0/before $R/AG/fast/PENG_JI_MING $R/AAA/ruputer/SU_KAI_LI; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  cat smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh 2>&1 | grep -v "^ *$" | grep -i "Reading\|Error\|error\|warning: \|libudf\|Done\|domain extents\|x-coordinate\|y-coordinate\|z-coordinate\|profile\|udf\|Checking mesh\|Mesh Check" | head -40
  echo "===== fluent exit code ${PIPESTATUS[0]} ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260921 2>/dev/null
done
