#!/bin/bash
#Author: XY
# ============================================================================
# 从 manifest.json 生成 Array Job 用的配置列表
# ============================================================================
# 使用方法:
#   ./generate_manifest_list.sh
#   ./generate_manifest_list.sh training/configs/field/generated/manifest.json baseline
#   ./generate_manifest_list.sh training/configs/field/generated/manifest.json baseline A-Main-01 2 4
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

MANIFEST_PATH=${1:-"training/configs/field/generated/manifest.json"}
STUDY_GROUP=${2:-""}
EXP_ID=${3:-""}
SEED=${4:-""}
LIMIT=${5:-"0"}
OUTPUT_FILE=${MANIFEST_LIST_FILE:-"$SCRIPT_DIR/manifest_list.tsv"}
TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

MANIFEST_ABS="$PROJECT_DIR/$MANIFEST_PATH"
if [ ! -f "$MANIFEST_ABS" ]; then
    MANIFEST_ABS="$MANIFEST_PATH"
fi

if [ ! -f "$MANIFEST_ABS" ]; then
    echo "错误: manifest 不存在: $MANIFEST_PATH"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"

if [ -n "$TRAINING_PYTHON" ]; then
    PYTHON_BIN="$TRAINING_PYTHON"
else
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
    else
        echo "错误: 找不到 conda 初始化脚本，请设置 TRAINING_PYTHON 或补充 module load anaconda3"
        exit 1
    fi
    conda activate "$TRAINING_ENV"
    PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
study_group = sys.argv[2]
exp_id = sys.argv[3]
seed_raw = sys.argv[4]
limit = int(sys.argv[5])
output_path = Path(sys.argv[6])

with manifest_path.open("r", encoding="utf-8") as f:
    items = list(json.load(f).get("items", []))

selected = []
for item in items:
    if study_group and item.get("study_group") != study_group:
        continue
    if exp_id and item.get("exp_id") != exp_id:
        continue
    if seed_raw and int(item.get("seed", -1)) != int(seed_raw):
        continue
    selected.append(item)

if limit > 0:
    selected = selected[:limit]

if not selected:
    raise SystemExit("筛选后没有可写入的实验项")

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    for item in selected:
        row = [
            str(item.get("exp_id", "")),
            str(item.get("study_group", "")),
            str(item.get("seed", "")),
            str(item["config_path"]),
        ]
        f.write("\t".join(row) + "\n")

print(f"已写入 {len(selected)} 条配置到: {output_path}")
' "$MANIFEST_ABS" "$STUDY_GROUP" "$EXP_ID" "$SEED" "$LIMIT" "$OUTPUT_FILE"

echo "前几条配置:"
sed -n '1,5p' "$OUTPUT_FILE"
