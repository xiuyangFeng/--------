#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?用法: bash submit_export_crown.sh <export_config.json>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
sbatch "${SCRIPT_DIR}/run_export_crown.slurm" "${CONFIG_PATH}"
