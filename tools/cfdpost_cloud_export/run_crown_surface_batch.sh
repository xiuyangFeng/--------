#!/usr/bin/env bash
# 批量：CROWN 三病例 merged-1146 可视化（与 V3P t016_report 对齐）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CROWN_CONFIG="${CROWN_CONFIG:-external_baselines/crown_beihang/configs/local/crown_original_vp_split_AG_v1_seed1.json}"
CROWN_CKPT="${CROWN_CKPT:-outputs/external_baselines/crown_beihang/crown_original_vp_split_AG_v1_seed1_20260619_162738/best_model.pt}"
SAMPLE_ID="${SAMPLE_ID:-result_features_merged-1146}"
RUN_TAG="${RUN_TAG:-crown_vp_t016_report}"
CROWN_METHOD_LABEL="${CROWN_METHOD_LABEL:-non-PINN}"
BUNDLE_ROOT="${BUNDLE_ROOT:-outputs/field/postview/${RUN_TAG}}"

export CROWN_CONFIG CROWN_CKPT SAMPLE_ID RUN_TAG CROWN_METHOD_LABEL BUNDLE_ROOT INFERENCE_MODE="${INFERENCE_MODE:-full}"

for CASE in "slow/GUO_XI_JIANG" "slow/ZHANG_JUN_HUA" "fast/CHEN_SHI_MING"; do
  echo "========== ${CASE} =========="
  CASE_NAME="${CASE}" bash tools/cfdpost_cloud_export/package_crown_postview_case.sh
done

echo "完成: ${BUNDLE_ROOT}"
