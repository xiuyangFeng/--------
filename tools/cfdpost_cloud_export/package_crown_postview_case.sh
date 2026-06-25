#!/usr/bin/env bash
# CROWN 单病例：推理导出 → Gaussian STL 映射 → 压力三联图 + ParaView 包
#
# 用法：
#   CROWN_CONFIG=external_baselines/crown_beihang/configs/local/crown_original_vp_split_AG_v1_seed1.json \
#   CROWN_CKPT=outputs/external_baselines/crown_beihang/.../best_model.pt \
#   CASE_NAME=slow/GUO_XI_JIANG \
#   SAMPLE_ID=result_features_merged-1146 \
#   bash tools/cfdpost_cloud_export/package_crown_postview_case.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CROWN_CONFIG="${CROWN_CONFIG:?需要 CROWN_CONFIG=...json}"
CROWN_CKPT="${CROWN_CKPT:?需要 CROWN_CKPT=.../best_model.pt}"
CASE_NAME="${CASE_NAME:?需要 CASE_NAME}"
SAMPLE_ID="${SAMPLE_ID:?需要 SAMPLE_ID}"
RUN_TAG="${RUN_TAG:-crown_vp_t016_report}"
CROWN_METHOD_LABEL="${CROWN_METHOD_LABEL:-non-PINN}"
INFERENCE_MODE="${INFERENCE_MODE:-full}"
METHOD="${METHOD:-gaussian}"
RADIUS="${RADIUS:-3.0}"
SHARPNESS="${SHARPNESS:-2.0}"
MAX_DIST="${MAX_DIST:-3.0}"
CHUNK_SIZE="${CHUNK_SIZE:-65536}"

CASE_SLUG="${CASE_NAME//\//__}"
CASE_SHORT="${CASE_NAME##*/}"
STEM="${CASE_SLUG}__${SAMPLE_ID}"
BUNDLE_ROOT="${BUNDLE_ROOT:-outputs/field/postview/${RUN_TAG}}"
CASE_DIR="${BUNDLE_ROOT}/${CASE_SHORT}__${SAMPLE_ID}"
EXPORT_DIR="${CASE_DIR}/_export"
SURFACE_DIR="${CASE_DIR}/surface_gaussian"
POINTCLOUD_DIR="${CASE_DIR}/pointcloud"
PLOT_DIR="${CASE_DIR}/plots"
STL_SRC="${STL_PATH:-data_new/AG/${CASE_NAME}/${CASE_SHORT}.stl}"

PYTHON="${PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}"

mkdir -p "${CASE_DIR}" "${EXPORT_DIR}" "${SURFACE_DIR}" "${POINTCLOUD_DIR}" "${PLOT_DIR}"

echo "========== 1/4 CROWN 推理 → 点云 CSV =========="
"${PYTHON}" "${SCRIPT_DIR}/export_crown_for_cfdpost.py" \
  --config "${CROWN_CONFIG}" \
  --checkpoint "${CROWN_CKPT}" \
  --case-name "${CASE_NAME}" \
  --sample-id "${SAMPLE_ID}" \
  --output-dir "${EXPORT_DIR}" \
  --inference-mode "${INFERENCE_MODE}" \
  --chunk-size "${CHUNK_SIZE}"

WALL_CSV="${EXPORT_DIR}/${STEM}__wall.csv"

echo "========== 2/4 Gaussian 插值 → STL 面片 VTP =========="
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
    --scalars "p_cfd,p_pred,err_p,abs_err_p,vel_mag_cfd,vel_mag_pred,err_vel_mag,abs_err_vel_mag,wss_cfd"
  cp -f "${SURFACE_DIR}/${STEM}__stl_mapped_wall.vtp" "${CASE_DIR}/${CASE_SHORT}__surface_wall.vtp"
  cp -f "${SURFACE_DIR}/${STEM}__stl_mapped_wall.csv" "${CASE_DIR}/${CASE_SHORT}__surface_wall.csv"
  cp -f "${SURFACE_DIR}/${STEM}__mapping_report_wall.json" "${CASE_DIR}/${CASE_SHORT}__mapping_report.json"
fi

echo "========== 3/4 点云 VTP + 压力三联图 =========="
"${PYTHON}" "${SCRIPT_DIR}/export_pointcloud_vtp.py" \
  --csv "${WALL_CSV}" \
  --output-dir "${POINTCLOUD_DIR}"
PC_VTP="${POINTCLOUD_DIR}/${STEM}__wall.vtp"
[[ -f "${PC_VTP}" ]] && cp -f "${PC_VTP}" "${CASE_DIR}/${CASE_SHORT}__pointcloud_wall.vtp"

VTP="${SURFACE_DIR}/${STEM}__stl_mapped_wall.vtp"
if [[ -f "${VTP}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
    --vtp "${VTP}" \
    --render surface \
    --variable p \
    --title "CROWN ${CROWN_METHOD_LABEL} · ${CASE_NAME} · ${SAMPLE_ID} · Pressure (${METHOD} r=${RADIUS})" \
    --output "${PLOT_DIR}/fig_p_triptych.png" \
    --field-cmap GNN_BWR \
    --err-cmap GNN_BWR \
    --report-json "${PLOT_DIR}/fig_p_triptych_report.json"

  "${PYTHON}" "${SCRIPT_DIR}/plot_stl_mapped_triptych.py" \
    --vtp "${VTP}" \
    --render surface \
    --variable vel_mag \
    --title "CROWN ${CROWN_METHOD_LABEL} · ${CASE_NAME} · ${SAMPLE_ID} · |Velocity| (${METHOD} r=${RADIUS})" \
    --output "${PLOT_DIR}/fig_vel_mag_triptych.png" \
    --field-cmap GNN_BWR \
    --err-cmap GNN_BWR \
    --report-json "${PLOT_DIR}/fig_vel_mag_triptych_report.json"
fi

echo "========== 4/4 复制配色与说明 =========="
cp -f "${SCRIPT_DIR}/paraview/GNN_blue_white_red.xml" "${CASE_DIR}/"
cp -f "${SCRIPT_DIR}/paraview/打开说明_ParaView.md" "${CASE_DIR}/README_后处理打开说明.md"

cat > "${CASE_DIR}/manifest_bundle.json" <<EOF
{
  "method": "CROWN/Beihang ${CROWN_METHOD_LABEL} (paper-original VP)",
  "case_name": "${CASE_NAME}",
  "sample_id": "${SAMPLE_ID}",
  "crown_config": "${CROWN_CONFIG}",
  "crown_checkpoint": "${CROWN_CKPT}",
  "inference_mode": "${INFERENCE_MODE}",
  "note": "CROWN 仅预测 u,v,w,p；无 WSS pred。壁面真值来自 features CSV，预测来自 CROWN full 点云推理+NN 映射（默认）。",
  "interpolation": {
    "method": "${METHOD}",
    "radius_mm": ${RADIUS},
    "sharpness": ${SHARPNESS},
    "max_dist_mm": ${MAX_DIST}
  },
  "files": {
    "surface_vtp_primary": "${CASE_SHORT}__surface_wall.vtp",
    "pressure_triptych": "plots/fig_p_triptych.png",
    "velocity_triptych": "plots/fig_vel_mag_triptych.png",
    "colormap": "GNN_blue_white_red.xml"
  }
}
EOF

echo ""
echo "✅ CROWN 病例包: ${CASE_DIR}"
echo "   ★ ParaView: ${CASE_SHORT}__surface_wall.vtp · 压力变量 p_cfd/p_pred/err_p"
ls -lh "${CASE_DIR}/"
