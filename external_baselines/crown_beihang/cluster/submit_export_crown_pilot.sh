#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-external_baselines/crown_beihang/configs/local/crown_export_split_AG_v1.json}"
PILOT_CASE="${2:-slow/LI_HUAN_GE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
sbatch "${SCRIPT_DIR}/run_export_crown_pilot.slurm" "${CONFIG_PATH}" "${PILOT_CASE}"
