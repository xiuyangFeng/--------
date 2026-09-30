#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=b3_smoke
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke_b3_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/smoke_b3_%j.err
#SBATCH --time=03:00:00
# 第三批冒烟（2026-09-22，13 例）：逐例只 read-case（触发 libudf 从协议修正后的 UDF 重编译）→ /mesh/check（域范围，验证 PENG_JI_MING 的 /mesh/scale）→ 退出，不迭代、不导出。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
for d in /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/PENG_JI_MING /public/newhome/cy/Digital_twin/GNN/data_new/ILO/AN_GUANG_JIE-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/GUO_AI_JUN-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/GUO_AI_JUN-0/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/GUO_QING_SHAN-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LI_FA_XIANG-1/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LI_YU_GANG-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/SHEN_CHUN_WANG-0/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/WANG_LI_MIN-0/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/XUE_YOU_TANG-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/XUE_YOU_TANG-0/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/ZHANG_MAO_JIN-0/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/ZHANG_WAN_ZENG-1/before ; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  cat smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh 2>&1 | grep -v "^ *$" | grep -i "Reading\|Error\|error\|warning: \|libudf\|Done\|domain extents\|x-coordinate\|y-coordinate\|z-coordinate\|profile\|udf\|Checking mesh\|Mesh Check" | head -40
  echo "===== fluent exit code ${PIPESTATUS[0]} ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260922 2>/dev/null
done
