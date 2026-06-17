#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-external_baselines/crown_beihang/configs/local/crown_export_split_AG_v1.json}"
MAX_PARALLEL="${2:-24}"
TASK_TIME="${3:-04:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$(dirname "$BASELINE_DIR")"
PROJECT_DIR="$(dirname "$EXTERNAL_DIR")"
CASES_FILE="${SCRIPT_DIR}/crown_export_cases_AG_v1.txt"

mkdir -p "${SCRIPT_DIR}/logs"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python -m external_baselines.crown_beihang.export_pkl \
    --config "$CONFIG_PATH" \
    --write-all-cases-file "$CASES_FILE"

N_CASES=$(grep -cve '^\s*$' "$CASES_FILE" || true)
N_CASES=$(grep -cve '^\s*#' "$CASES_FILE" || echo "$N_CASES")
# 排除注释行
N_CASES=$(grep -v '^\s*#' "$CASES_FILE" | grep -cve '^\s*$' || true)
LAST_IDX=$((N_CASES - 1))

echo "Cases: ${N_CASES} · array 0-${LAST_IDX}%${MAX_PARALLEL} · time=${TASK_TIME}"

ARRAY_JOB=$(sbatch --parsable \
    --array="0-${LAST_IDX}%${MAX_PARALLEL}" \
    --time="${TASK_TIME}" \
    "${SCRIPT_DIR}/run_export_crown_array.slurm" \
    "${CONFIG_PATH}" \
    "${CASES_FILE}")

echo "Array export Job ${ARRAY_JOB}"

MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    "${SCRIPT_DIR}/run_export_crown_merge.slurm" \
    "${CONFIG_PATH}")

echo "Merge Job ${MERGE_JOB} (afterok:${ARRAY_JOB})"
