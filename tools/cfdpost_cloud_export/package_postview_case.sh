#!/usr/bin/env bash
# 打包单个病例 → 后处理软件可旋转查看的交付目录（ParaView / CFD-Post）
#
# 用法：
#   MANIFEST=outputs/field/.../predictions_test_best_wss/manifest.json \
#   CASE_NAME=slow/GUO_XI_JIANG \
#   SAMPLE_ID=result_features_merged-1120 \
#   bash tools/cfdpost_cloud_export/package_postview_case.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:?需要 MANIFEST=.../manifest.json}"
CASE_NAME="${CASE_NAME:?需要 CASE_NAME}"
SAMPLE_ID="${SAMPLE_ID:?需要 SAMPLE_ID}"
RUN_TAG="${RUN_TAG:-v3p_postview}"
METHOD="${METHOD:-gaussian}"
RADIUS="${RADIUS:-3.0}"
SHARPNESS="${SHARPNESS:-2.0}"
MAX_DIST="${MAX_DIST:-3.0}"

CASE_SLUG="${CASE_NAME//\//__}"
CASE_SHORT="${CASE_NAME##*/}"
STEM="${CASE_SLUG}__${SAMPLE_ID}"
BUNDLE_ROOT="${BUNDLE_ROOT:-outputs/field/postview/${RUN_TAG}}"
CASE_DIR="${BUNDLE_ROOT}/${CASE_SHORT}__${SAMPLE_ID}"
EXPORT_DIR="${CASE_DIR}/_export"
SURFACE_DIR="${CASE_DIR}/surface_gaussian"
POINTCLOUD_DIR="${CASE_DIR}/pointcloud"
STL_SRC="${STL_PATH:-data_new/AG/${CASE_NAME}/${CASE_SHORT}.stl}"

PYTHON="${PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

mkdir -p "${CASE_DIR}" "${EXPORT_DIR}" "${SURFACE_DIR}" "${POINTCLOUD_DIR}"

echo "========== 1/4 导出点云 CSV =========="
"${PYTHON}" "${SCRIPT_DIR}/export_for_cfdpost.py" \
  --manifest "${MANIFEST}" \
  --sample-id "${SAMPLE_ID}" \
  --case-name "${CASE_NAME}" \
  --output-dir "${EXPORT_DIR}"

WALL_CSV="${EXPORT_DIR}/${STEM}__wall.csv"
ALL_CSV="${EXPORT_DIR}/${STEM}__all.csv"

echo "========== 2/4 Gaussian 插值 → STL 面片 VTP（★ ParaView 旋转主文件） =========="
if [[ ! -f "${STL_SRC}" ]]; then
  echo "[warn] STL 不存在: ${STL_SRC}，跳过面片映射"
else
  cp -f "${STL_SRC}" "${CASE_DIR}/${CASE_SHORT}.stl"
  "${PYTHON}" "${SCRIPT_DIR}/map_to_stl_surface.py" \
    --csv "${WALL_CSV}" \
    --stl "${STL_SRC}" \
    --output-dir "${SURFACE_DIR}" \
    --method "${METHOD}" \
    --max-dist "${MAX_DIST}" \
    --radius "${RADIUS}" \
    --sharpness "${SHARPNESS}" \
    --scalars "wss_cfd,wss_pred,err_wss,abs_err_wss,p_cfd,p_pred,err_p,abs_err_p"
  cp -f "${SURFACE_DIR}/${STEM}__stl_mapped_wall.vtp" "${CASE_DIR}/${CASE_SHORT}__surface_wall.vtp"
  cp -f "${SURFACE_DIR}/${STEM}__stl_mapped_wall.csv" "${CASE_DIR}/${CASE_SHORT}__surface_wall.csv"
  cp -f "${SURFACE_DIR}/${STEM}__mapping_report_wall.json" "${CASE_DIR}/${CASE_SHORT}__mapping_report.json"
fi

echo "========== 3/4 点云 VTP（补充：看原始采样点分布） =========="
"${PYTHON}" "${SCRIPT_DIR}/export_pointcloud_vtp.py" \
  --csv "${WALL_CSV}" \
  --output-dir "${POINTCLOUD_DIR}"
PC_VTP="${POINTCLOUD_DIR}/${STEM}__wall.vtp"
[[ -f "${PC_VTP}" ]] && cp -f "${PC_VTP}" "${CASE_DIR}/${CASE_SHORT}__pointcloud_wall.vtp"

echo "========== 4/4 复制配色与说明 =========="
cp -f "${SCRIPT_DIR}/paraview/GNN_blue_white_red.xml" "${CASE_DIR}/"
cp -f "${SCRIPT_DIR}/paraview/打开说明_ParaView.md" "${CASE_DIR}/README_后处理打开说明.md"

cat > "${CASE_DIR}/manifest_bundle.json" <<EOF
{
  "case_name": "${CASE_NAME}",
  "sample_id": "${SAMPLE_ID}",
  "gnn_manifest": "${MANIFEST}",
  "interpolation": {
    "method": "${METHOD}",
    "radius_mm": ${RADIUS},
    "sharpness": ${SHARPNESS},
    "max_dist_mm": ${MAX_DIST}
  },
  "files": {
    "surface_vtp_primary": "${CASE_SHORT}__surface_wall.vtp",
    "surface_csv": "${CASE_SHORT}__surface_wall.csv",
    "pointcloud_vtp": "${CASE_SHORT}__pointcloud_wall.vtp",
    "stl_geometry": "${CASE_SHORT}.stl",
    "colormap": "GNN_blue_white_red.xml",
    "mapping_report": "${CASE_SHORT}__mapping_report.json"
  },
  "paraview_hint": "Open surface_vtp → Representation=Surface → Color=wss_cfd|wss_pred|err_wss → drag to rotate",
  "cfdpost_hint": "Import surface_csv as point cloud on STL, or use Fluent .cas + CSV overlay"
}
EOF

echo ""
echo "✅ 病例包已生成: ${CASE_DIR}"
echo "   ★ ParaView 打开: ${CASE_SHORT}__surface_wall.vtp"
ls -lh "${CASE_DIR}/"
