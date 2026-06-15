#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-external_baselines/pointnetcfd/configs}"

for config_path in "${CONFIG_DIR}"/pointnetcfd_*.json; do
  echo "submit ${config_path}"
  bash external_baselines/pointnetcfd/cluster/submit_pointnetcfd.sh "${config_path}"
done
