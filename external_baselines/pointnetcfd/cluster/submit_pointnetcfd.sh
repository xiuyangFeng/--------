#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?用法: bash external_baselines/pointnetcfd/cluster/submit_pointnetcfd.sh <config.json>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
sbatch "${SCRIPT_DIR}/run_pointnetcfd.slurm" "${CONFIG_PATH}"
