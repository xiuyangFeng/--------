#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=syn_smoke4
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/smoke_synth_regen_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926/smoke_synth_regen_%j.err
#SBATCH --time=06:00:00
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
for d in /public/newhome/cy/Digital_twin/GNN/data_new/AAA/unruputer/HAN_JIAN_FU~m36; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh 2>&1 | grep -v "^ *$" | grep -i "Reading\|Error\|error\|warning: \|libudf\|Done\|domain extents\|coordinate\|Checking mesh\|Mesh Check\|quality\|skew" | head -60
  echo "===== fluent exit code ${PIPESTATUS[0]} ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260927 2>/dev/null
done
