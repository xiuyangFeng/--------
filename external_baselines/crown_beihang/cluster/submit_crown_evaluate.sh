#!/usr/bin/env bash
# 集群提交 CROWN evaluate（勿在 Cursor / 登录节点 nohup 长跑）
#
# 用法:
#   ./submit_crown_evaluate.sh <checkpoint> [--split test] [--eval-mode paper|subsample] [--log-every N] [--output-dir DIR]
#
# 示例（5751 paper 全点）:
#   ./submit_crown_evaluate.sh \
#     outputs/external_baselines/crown_beihang/crown_original_vp_split_AG_v1_seed1_20260619_162738/best_model.pt \
#     --eval-mode paper --log-every 20
set -euo pipefail

CHECKPOINT_PATH="${1:?usage: submit_crown_evaluate.sh <checkpoint> [--split test] [--eval-mode paper|subsample] ...}"
shift

SPLIT_NAME="test"
EVAL_MODE="paper"
LOG_EVERY="20"
OUTPUT_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --split)
            SPLIT_NAME="${2:?missing value for --split}"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="${2:?missing value for --eval-mode}"
            shift 2
            ;;
        --log-every)
            LOG_EVERY="${2:?missing value for --log-every}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
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

SBATCH_BIN="${SBATCH_BIN:-/public/slurm/bin/sbatch}"
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1 && [ -x /public/slurm/bin/sbatch ]; then
    SBATCH_BIN="/public/slurm/bin/sbatch"
fi

echo "submit CROWN evaluate: checkpoint=${CHECKPOINT_PATH} split=${SPLIT_NAME} mode=${EVAL_MODE} log_every=${LOG_EVERY}"
"$SBATCH_BIN" "${SCRIPT_DIR}/run_crown_evaluate.slurm" \
    "$CHECKPOINT_PATH" "$SPLIT_NAME" "$EVAL_MODE" "$LOG_EVERY" "$OUTPUT_DIR"
