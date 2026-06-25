#!/usr/bin/env bash
# 单病例：预测 CSV 导出 → STL 映射 → WSS/P 三联图
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:?需要 MANIFEST=.../predictions_test_best_wss/manifest.json}"
SAMPLE_ID="${SAMPLE_ID:?需要 SAMPLE_ID}"
CASE_NAME="${CASE_NAME:?需要 CASE_NAME，如 slow/GUO_XI_JIANG}"
RUN_TAG="${RUN_TAG:-post5463_i6diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/field/plots/stl_surface_compare/${RUN_TAG}}"
EXPORT_DIR="${EXPORT_DIR:-tools/cfdpost_cloud_export/output/${RUN_TAG}}"
MAX_DIST="${MAX_DIST:-3.0}"
METHOD="${METHOD:-gaussian}"
RADIUS="${RADIUS:-3.0}"
SHARPNESS="${SHARPNESS:-2.0}"
FIELD_CMAP="${FIELD_CMAP:-GNN_BWR}"
ERR_CMAP="${ERR_CMAP:-GNN_BWR}"

CASE_SLUG="${CASE_NAME//\//__}"
STEM="${CASE_SLUG}__${SAMPLE_ID}"
WALL_CSV="${EXPORT_DIR}/${STEM}__wall.csv"
STL_PATH="${STL_PATH:-data_new/AG/${CASE_NAME}/${CASE_NAME##*/}.stl}"

PYTHON="${PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
PYTHON_VTK="${PYTHON_VTK:-/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python}"

mkdir -p "${OUTPUT_ROOT}" "${EXPORT_DIR}/route_interp"

echo "========== 导出点云 CSV =========="
"${PYTHON}" "${SCRIPT_DIR}/export_for_cfdpost.py" \
  --manifest "${MANIFEST}" \
  --sample-id "${SAMPLE_ID}" \
  --case-name "${CASE_NAME}" \
  --output-dir "${EXPORT_DIR}"

if [[ ! -f "${STL_PATH}" ]]; then
  echo "STL 不存在: ${STL_PATH}" >&2
  exit 1
fi

echo "========== 映射到 STL 面片 (${METHOD}) =========="
"${PYTHON}" "${SCRIPT_DIR}/map_to_stl_surface.py" \
  --csv "${WALL_CSV}" \
  --stl "${STL_PATH}" \
  --output-dir "${EXPORT_DIR}/route_interp" \
  --method "${METHOD}" \
  --max-dist "${MAX_DIST}" \
  --radius "${RADIUS}" \
  --sharpness "${SHARPNESS}" \
  --scalars "wss_cfd,wss_pred,err_wss,abs_err_wss,p_cfd,p_pred,err_p"

VTP="${EXPORT_DIR}/route_interp/${STEM}__stl_mapped_wall.vtp"
CSV="${EXPORT_DIR}/route_interp/${STEM}__stl_mapped_wall.csv"
CASE_OUT="${OUTPUT_ROOT}/${STEM}"
mkdir -p "${CASE_OUT}"

echo "========== 生成 WSS 三联图（STL 三角面片） =========="
"${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
  --vtp "${VTP}" \
  --render surface \
  --variable wss \
  --title "${CASE_NAME} · ${SAMPLE_ID} · WSS (${METHOD} r=${RADIUS} sh=${SHARPNESS}mm)" \
  --output "${CASE_OUT}/fig_wss_triptych.png" \
  --field-cmap "${FIELD_CMAP}" \
  --err-cmap "${ERR_CMAP}" \
  --report-json "${CASE_OUT}/fig_wss_triptych_report.json"

echo "========== 生成压力三联图（STL 三角面片） =========="
"${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
  --vtp "${VTP}" \
  --render surface \
  --variable p \
  --title "${CASE_NAME} · ${SAMPLE_ID} · Pressure (${METHOD} r=${RADIUS} sh=${SHARPNESS}mm)" \
  --output "${CASE_OUT}/fig_p_triptych.png" \
  --field-cmap "${FIELD_CMAP}" \
  --err-cmap "${ERR_CMAP}" \
  --report-json "${CASE_OUT}/fig_p_triptych_report.json"

echo "完成: ${CASE_OUT}"
