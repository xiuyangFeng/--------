#!/usr/bin/env bash
# 一键导出 GNN 预测为 CFD-Post 点云 CSV（示例：GUO_XI_JIANG / merged-1120）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 可通过环境变量覆盖默认值
MANIFEST="${MANIFEST:-outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260609_124213/predictions_test_best_wss/manifest.json}"
SAMPLE_ID="${SAMPLE_ID:-result_features_merged-1120}"
CASE_NAME="${CASE_NAME:-slow/GUO_XI_JIANG}"
OUTPUT_DIR="${OUTPUT_DIR:-tools/cfdpost_cloud_export/output}"
NORM_PARAMS="${NORM_PARAMS:-data_new/AG/normalization_params_global.json}"

echo "仓库根目录: ${REPO_ROOT}"
echo "manifest:   ${MANIFEST}"
echo "样本:       ${SAMPLE_ID}"
echo "病例:       ${CASE_NAME}"
echo "输出目录:   ${OUTPUT_DIR}"
echo ""

python "${SCRIPT_DIR}/export_for_cfdpost.py" \
  --manifest "${MANIFEST}" \
  --sample-id "${SAMPLE_ID}" \
  --case-name "${CASE_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --norm-params "${NORM_PARAMS}"
