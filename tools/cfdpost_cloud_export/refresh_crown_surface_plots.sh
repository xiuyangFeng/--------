#!/usr/bin/env bash
# 已有 CROWN _export CSV 时，仅重跑 STL 映射 + 压力/速度三联图（跳过 GPU 推理）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-crown_vp_t016_report}"
BUNDLE_ROOT="${BUNDLE_ROOT:-outputs/field/postview/${RUN_TAG}}"
METHOD="${METHOD:-gaussian}"
RADIUS="${RADIUS:-3.0}"
SHARPNESS="${SHARPNESS:-2.0}"
MAX_DIST="${MAX_DIST:-3.0}"
SAMPLE_ID="${SAMPLE_ID:-result_features_merged-1146}"

SCRIPT_DIR="tools/cfdpost_cloud_export"
PYTHON="${PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

for CASE in "slow/GUO_XI_JIANG" "slow/ZHANG_JUN_HUA" "fast/CHEN_SHI_MING"; do
  CASE_SHORT="${CASE##*/}"
  CASE_SLUG="${CASE//\//__}"
  STEM="${CASE_SLUG}__${SAMPLE_ID}"
  CASE_DIR="${BUNDLE_ROOT}/${CASE_SHORT}__${SAMPLE_ID}"
  WALL_CSV="${CASE_DIR}/_export/${STEM}__wall.csv"
  SURFACE_DIR="${CASE_DIR}/surface_gaussian"
  PLOT_DIR="${CASE_DIR}/plots"
  STL_SRC="data_new/AG/${CASE}/${CASE_SHORT}.stl"
  VTP="${SURFACE_DIR}/${STEM}__stl_mapped_wall.vtp"

  echo "========== ${CASE} · remap + plots =========="
  [[ -f "${WALL_CSV}" ]] || { echo "缺少 ${WALL_CSV}" >&2; exit 1; }

  "${PYTHON}" "${SCRIPT_DIR}/map_to_stl_surface.py" \
    --csv "${WALL_CSV}" \
    --stl "${STL_SRC}" \
    --output-dir "${SURFACE_DIR}" \
    --method "${METHOD}" \
    --max-dist "${MAX_DIST}" \
    --radius "${RADIUS}" \
    --sharpness "${SHARPNESS}" \
    --scalars "p_cfd,p_pred,err_p,abs_err_p,vel_mag_cfd,vel_mag_pred,err_vel_mag,abs_err_vel_mag,wss_cfd"

  cp -f "${VTP}" "${CASE_DIR}/${CASE_SHORT}__surface_wall.vtp"
  cp -f "${SURFACE_DIR}/${STEM}__stl_mapped_wall.csv" "${CASE_DIR}/${CASE_SHORT}__surface_wall.csv"

  mkdir -p "${PLOT_DIR}"

  "${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
    --vtp "${VTP}" --render surface --variable p \
    --title "CROWN non-PINN · ${CASE} · ${SAMPLE_ID} · Pressure (${METHOD} r=${RADIUS})" \
    --output "${PLOT_DIR}/fig_p_triptych.png" \
    --field-cmap GNN_BWR --err-cmap GNN_BWR \
    --report-json "${PLOT_DIR}/fig_p_triptych_report.json"

  "${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
    --vtp "${VTP}" --render surface --variable vel_mag \
    --title "CROWN non-PINN · ${CASE} · ${SAMPLE_ID} · |Velocity| (${METHOD} r=${RADIUS})" \
    --output "${PLOT_DIR}/fig_vel_mag_triptych.png" \
    --field-cmap GNN_BWR --err-cmap GNN_BWR \
    --report-json "${PLOT_DIR}/fig_vel_mag_triptych_report.json"
  echo "完成: ${PLOT_DIR}"
done

echo "全部完成: ${BUNDLE_ROOT}"
