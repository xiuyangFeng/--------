#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=v52d_wtq_cl
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/wtq_centerline_%j.out
# 2026-10-01 merge: WANG_TIAN_QING-1/after was rebuilt from its own STL (new mesh). Its old centreline / atlas describe the
# DONG_JIA_JU mesh: moved aside first, then centreline (mesh wall, vmtk) and feature atlas for the new mesh.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python; PYV=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
U=ILO/WANG_TIAN_QING-1/after
CL=$G/outputs/centerline_v2_meshwall_20261001_merge
PREP=$G/outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$G"
date -Is; hostname
for old in "$G/outputs/centerline_v2_meshwall_20260922_new95/cases/$U" "$PREP/atlas/cases/$U"; do
  if [ -d "$old" ]; then dst="$(dirname "$old")/_stale_merge_20261001_$(basename "$old")"; [ -e "$dst" ] || mv "$old" "$dst"; echo "moved aside: $old"; fi; done
echo "== 1 centerline"; $PYV tools/centerline_v2_meshwall.py --cases $U --output "$CL"
echo "== 2 feature atlas"; $PY -m wss_pinn.v4.build_feature_atlas --cases $U --extra-cases $U --source-root "$CL" --workers 4
ls -la "$PREP/atlas/cases/$U"
echo "== done"; date -Is
