#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=peng_scale
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/peng_scale_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/peng_scale_%j.err
#SBATCH --time=01:00:00
# PENG_JI_MING：cas 网格比真实小 1000 倍（运行时 journal /mesh/scale 1000 放大后再算、导出为放大后坐标）。为让审计/入库链与其余 170 例走完全相同的代码路径，
# 用 Fluent 读入 → /mesh/scale 1000 → 写出 PENG_JI_MING_scaled.cas.gz（不迭代、不导出）；随后 2.jou 改读 scaled 文件并删掉 /mesh/scale 行。
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
cd /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/PENG_JI_MING || exit 1
mv -f libudf libudf_scale_20260922 2>/dev/null
cat > scale_write.jou <<'J'
/file/read-case /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/PENG_JI_MING/PENG_JI_MING.cas.gz
/mesh/scale 1000 1000 1000
/mesh/check
/file/write-case /public/newhome/cy/Digital_twin/GNN/data_new/AG/fast/PENG_JI_MING/PENG_JI_MING_scaled.cas.gz
/exit y
J
fluent 3ddp -g -t8 -mpi=intel -platform=intel -i scale_write.jou -ssh 2>&1 | grep -i "Reading\|Writing\|Error\|Domain Extents\|coordinate\|Done" | head -30
echo "exit ${PIPESTATUS[0]}"; ls -la PENG_JI_MING_scaled.cas.gz; rm -f scale_write.jou *.trn cleanup-fluent-*.sh; mv -f libudf libudf_scale_20260922b 2>/dev/null; true
