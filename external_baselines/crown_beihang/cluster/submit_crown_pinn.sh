#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-external_baselines/crown_beihang/configs/local/crown_original_vp_pinn_split_AG_v1_seed1.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

SBATCH_BIN="${SBATCH_BIN:-/public/slurm/bin/sbatch}"
if ! command -v "${SBATCH_BIN}" >/dev/null 2>&1 && [ -x /public/slurm/bin/sbatch ]; then
  SBATCH_BIN=/public/slurm/bin/sbatch
fi

echo "submit PINN lazy train: ${CONFIG_PATH}"
"${SBATCH_BIN}" "${SCRIPT_DIR}/run_crown_pinn.slurm" "${CONFIG_PATH}"
