#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH -w node06
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=udf_smoke
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/WSS_PINN/RCR出口面积录入核查_2026-09-15/smoke_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/WSS_PINN/RCR出口面积录入核查_2026-09-15/smoke_%j.err
# 冒烟测试：只 read-case（触发 libudf 自动编译）然后退出，不迭代。验证改过路径的 .cas.gz 能读、改过的 UDF 能编译。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
for d in /public/newhome/cy/Digital_twin/GNN/data_new/AAA/ruputer/LIU_YONG_LAN /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/WANG_CHUN_MING; do
  cd "$d" || exit 1
  echo "===== SMOKE $d ====="
  { head -1 2.jou; echo "/exit y"; } > smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh
  echo "===== fluent exit code $? ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
done
