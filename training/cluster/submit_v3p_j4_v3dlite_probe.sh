#!/usr/bin/env bash
# V3P-J4 · V3D-lite MixedProbe seed1 短训（仅 Idea G oracle Go 后手动提交）
# 前置：python -m training.scripts.build_v3p_v3dlite_split
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${1:-training/configs/field/generated/v3_pointcloud/V3P-J4-V3Dlite-MixedProbe-post5463_seed1.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "缺少配置: $CONFIG"
  echo "请先运行 build_v3p_v3dlite_split 并确认 Idea G gate Go"
  exit 1
fi

if [[ ! -f training/splits/split_AG_v3lite_v1.json ]]; then
  echo "生成 split ..."
  /public/newhome/cy/.conda/envs/GNN/bin/python -m training.scripts.build_v3p_v3dlite_split
fi

echo "提交 J4 V3D-lite MixedProbe: $CONFIG"
sbatch training/cluster/run_train_field.slurm "$CONFIG"
