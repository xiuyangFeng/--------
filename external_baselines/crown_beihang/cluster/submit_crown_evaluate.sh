#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?usage: submit_crown_evaluate.sh <checkpoint> [--split test]}"
shift

SPLIT_NAME="test"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --split)
            SPLIT_NAME="${2:?missing value for --split}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1 && [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
fi

"$SBATCH_BIN" "${SCRIPT_DIR}/run_crown_evaluate.slurm" "$CHECKPOINT_PATH" "$SPLIT_NAME"
