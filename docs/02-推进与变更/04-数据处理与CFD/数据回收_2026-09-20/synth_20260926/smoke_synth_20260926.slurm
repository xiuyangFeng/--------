#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=syn_smoke
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/smoke_synth_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/smoke_synth_%j.err
#SBATCH --time=06:00:00
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
for d in /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/PENG_JI_MING~m25 /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/KANG_XI_MING~m18 /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LI_BING_YI~m42 /public/newhome/cy/Digital_twin/GNN/data_new/AAA/ruputer/LV_GUO_YOU~m32 /public/newhome/cy/Digital_twin/GNN/data_new/AAA/unruputer/CHEN_SHU_LIN~m25 /public/newhome/cy/Digital_twin/GNN/data_new/ILO/ZHANG_YONG_SHENG-0~m52/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LU_FU_SHAN-0~m21/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/WANG_JIN_MING-0~m54/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/AN_GUANG_JIE-0~m37/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LI_FA_XIANG-1~m42/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LI_YU_GANG-0~m01/after /public/newhome/cy/Digital_twin/GNN/data_new/ILO/WANG_SHU_SHENG-0~m36/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/XUE_YOU_TANG-0~m19/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/PAN_YIN_GEN-1~m45/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LIU_YUE_DONG-0~m29/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/ZHAO_CHANG_SHAN-0~m06/before /public/newhome/cy/Digital_twin/GNN/data_new/ILO/LIU_BAO_JUN-0~m39/before ; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh 2>&1 | grep -v "^ *$" | grep -i "Reading\|Error\|error\|warning: \|libudf\|Done\|domain extents\|coordinate\|Checking mesh\|Mesh Check\|quality\|skew" | head -60
  echo "===== fluent exit code ${PIPESTATUS[0]} ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260926 2>/dev/null
done
