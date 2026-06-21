#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=hou_reextract
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00

# HOU_SHEN_QIAN: STL 已按 1/1000 缩放对齐 CFD 点云，重跑 step2→local_wss→step3-5
set -euo pipefail

CASE="HOU_SHEN_QIAN"
DATA_ROOT="data_new/AG"
SOURCES="slow"
TRAIN_SPLIT="training/splits/split_AG_v1.json"
GRAPHS_SUBDIR="processed/graphs_local_v1"

module purge
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
source "$(conda info --base)/etc/profile.d/conda.sh"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs pipeline/cluster/logs

echo "=== HOU_SHEN_QIAN 中心线重处理 $(date) ==="
echo "DATA_ROOT=$DATA_ROOT CASE=$CASE"

conda activate GNN_vmtk
export PIPELINE_DATA_ROOT="$DATA_ROOT"
echo ">>> step2 extract_features"
python -m pipeline.extract_features --case "$CASE" --sources "$SOURCES"

conda activate GNN
echo ">>> local_wss"
python -m pipeline.local_wss --data-root "$DATA_ROOT" --sources "$SOURCES" --case "$CASE" --normal-source stl

echo ">>> step3 coord_normalize"
python -m pipeline.coord_normalize --data-root "$DATA_ROOT" --sources "$SOURCES" --case "$CASE"

echo ">>> step4 normalize"
python -m pipeline.normalize \
  --data-root "$DATA_ROOT" \
  --train-split "${PROJECT_DIR}/${TRAIN_SPLIT}" \
  --sources "$SOURCES" \
  --case "$CASE"

echo ">>> step5 convert_to_graph"
python -m pipeline.convert_to_graph \
  --data-root "$DATA_ROOT" \
  --output-subdir "$GRAPHS_SUBDIR" \
  --sources "$SOURCES" \
  --case "$CASE"

echo "=== 完成 $(date) ==="
