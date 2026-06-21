#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-external_baselines/crown_beihang/configs/local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

SBATCH_EXTRA=()
if [ -n "${DEPENDENCY:-}" ]; then
  SBATCH_EXTRA=(--dependency="${DEPENDENCY}")
fi

for config_path in "${CONFIG_DIR}"/crown_original_vp_*.json; do
  case "$config_path" in
    *export*) continue ;;
  esac
  echo "submit ${config_path} ${SBATCH_EXTRA[*]:-}"
  sbatch "${SBATCH_EXTRA[@]}" --partition=GPU \
    "${SCRIPT_DIR}/run_crown.slurm" "${config_path}"
done
