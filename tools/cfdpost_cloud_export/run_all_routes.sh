#!/usr/bin/env bash
# 一键：导出 CSV + 路线 B（STL 映射 + Fluent 插值 CSV）+ 路线 C（VTP 点云）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:-outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260609_124213/predictions_test_best_wss/manifest.json}"
SAMPLE_ID="${SAMPLE_ID:-result_features_merged-1120}"
CASE_NAME="${CASE_NAME:-slow/GUO_XI_JIANG}"
OUTPUT_DIR="${OUTPUT_DIR:-tools/cfdpost_cloud_export/output}"
NORM_PARAMS="${NORM_PARAMS:-data_new/AG/normalization_params_global.json}"
STL_PATH="${STL_PATH:-data_new/AG/slow/GUO_XI_JIANG/GUO_XI_JIANG.stl}"
MAX_DIST="${MAX_DIST:-3.0}"
SKIP_STL="${SKIP_STL:-0}"

CASE_SLUG="${CASE_NAME//\//__}"
STEM="${CASE_SLUG}__${SAMPLE_ID}"
WALL_CSV="${OUTPUT_DIR}/${STEM}__wall.csv"
INTERIOR_CSV="${OUTPUT_DIR}/${STEM}__interior.csv"

PYTHON="${PYTHON:-python}"
PYTHON_VTK="${PYTHON_VTK:-}"
if [[ -z "${PYTHON_VTK}" ]] && command -v conda &>/dev/null; then
  if conda run -n GNN_vmtk python -c "import vtk" &>/dev/null; then
    PYTHON_VTK="conda run -n GNN_vmtk python"
  fi
fi
[[ -z "${PYTHON_VTK}" ]] && PYTHON_VTK="${PYTHON}"

echo "仓库根目录: ${REPO_ROOT}"
echo "CSV 导出 Python: ${PYTHON}"
echo "VTK  脚本 Python: ${PYTHON_VTK}"

echo ""
echo "========== 步骤 0/4: 导出点云 CSV（公共） =========="
"${PYTHON}" "${SCRIPT_DIR}/export_for_cfdpost.py" \
  --manifest "${MANIFEST}" \
  --sample-id "${SAMPLE_ID}" \
  --case-name "${CASE_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --norm-params "${NORM_PARAMS}"

echo ""
echo "========== 步骤 1/4: 路线 B1 — 映射到 STL 面片 =========="
if [[ "${SKIP_STL}" == "1" ]]; then
  echo "SKIP_STL=1，跳过 STL 映射"
else
  if [[ ! -f "${REPO_ROOT}/${STL_PATH}" ]] && [[ ! -f "${STL_PATH}" ]]; then
    echo "[warn] 未找到 STL: ${STL_PATH}，跳过 B1。可设置 STL_PATH 或 SKIP_STL=1"
  else
    STL="${STL_PATH}"
    [[ -f "${REPO_ROOT}/${STL_PATH}" ]] && STL="${REPO_ROOT}/${STL_PATH}"
    ${PYTHON_VTK} "${SCRIPT_DIR}/map_to_stl_surface.py" \
      --csv "${WALL_CSV}" \
      --stl "${STL}" \
      --output-dir "${OUTPUT_DIR}/route_interp" \
      --max-dist "${MAX_DIST}"
    ${PYTHON_VTK} "${SCRIPT_DIR}/map_to_stl_surface.py" \
      --csv "${INTERIOR_CSV}" \
      --stl "${STL}" \
      --output-dir "${OUTPUT_DIR}/route_interp" \
      --max-dist "${MAX_DIST}" \
      --scalars "p_cfd,p_pred,err_p" || true
  fi
fi

echo ""
echo "========== 步骤 2/4: 路线 B2 — Fluent 点云插值 CSV =========="
"${PYTHON}" "${SCRIPT_DIR}/prepare_fluent_interpolation.py" \
  --csv "${WALL_CSV}" \
  --output-dir "${OUTPUT_DIR}/route_interp/fluent_cloud"
"${PYTHON}" "${SCRIPT_DIR}/prepare_fluent_interpolation.py" \
  --csv "${INTERIOR_CSV}" \
  --output-dir "${OUTPUT_DIR}/route_interp/fluent_cloud" \
  --fields "p_cfd,p_pred"

echo ""
echo "========== 步骤 3/4: 路线 C — 点云 VTP =========="
${PYTHON_VTK} "${SCRIPT_DIR}/export_pointcloud_vtp.py" \
  --csv "${WALL_CSV}" \
  --output-dir "${OUTPUT_DIR}/route_pointcloud"
${PYTHON_VTK} "${SCRIPT_DIR}/export_pointcloud_vtp.py" \
  --csv "${INTERIOR_CSV}" \
  --output-dir "${OUTPUT_DIR}/route_pointcloud"

echo ""
echo "========== 完成 =========="
echo "路线 A: 请在 Fluent 导出 .dat → tools/cfdpost_cloud_export/fluent_native/"
echo "路线 B: ${OUTPUT_DIR}/route_interp/"
echo "路线 C: ${OUTPUT_DIR}/route_pointcloud/"
echo "说明:   ${SCRIPT_DIR}/三条对比路线.md"
