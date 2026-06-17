#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-external_baselines/crown_beihang/configs/local}"

for config_path in "${CONFIG_DIR}"/crown_original_vp_*.json; do
  case "$config_path" in
    *export*) continue ;;
  esac
  echo "submit ${config_path}"
  bash external_baselines/crown_beihang/cluster/submit_crown.sh "${config_path}"
done
