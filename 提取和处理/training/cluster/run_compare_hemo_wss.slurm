#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=field_hemo_wss
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
# 集群规范：CPU 任务指定 node03（若排队过久可临时注释本行）
#SBATCH -w node03

# ============================================================================
# 任务 A / Line W：多 run 壁面 WSS 对比（compare_hemo_wss_runs）
#
# 依赖：各 run 已执行 predict_field，生成 predictions_*/manifest.json。
# 环境：GNN（torch / scipy）；纯 CPU。
#
# 用法（在仓库根目录 GNN 下提交）:
#   sbatch training/cluster/run_compare_hemo_wss.slurm
#   sbatch training/cluster/run_compare_hemo_wss.slurm training/cluster/hemo_wss_runs_A_Opt03_04_05_seed1.tsv
#
# 环境变量（可选）:
#   HEMO_OUTPUT_DIR   结果目录，相对仓库根；默认 outputs/field/plots/optimization/wss_compare_<JOBID>
#   HEMO_MAX_ITEMS    只处理每个 manifest 前 N 条（试跑）；不设置或 0 表示全量
#   HEMO_NO_EXPORT=1  不调用 export_hemo（假定 hemo_ai/hemo_cfd 已存在，仅汇总）
#   TRAINING_ENV      默认 GNN
#   TRAINING_PYTHON   若设置则优先于 conda 环境
#
# 日志: training/cluster/logs/field_hemo_wss_<JOBID>.out / .err
# ============================================================================

set -euo pipefail

RUNS_TSV=${1:-training/cluster/hemo_wss_runs_A_Opt03_04_05_seed1.tsv}

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
if [ -n "$TRAINING_PYTHON" ]; then
    PYTHON_BIN="$TRAINING_PYTHON"
else
    conda activate "$TRAINING_ENV"
    PYTHON_BIN="$(command -v python)"
fi

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

RUNS_TSV_ABS="$PROJECT_DIR/$RUNS_TSV"
if [ ! -f "$RUNS_TSV_ABS" ]; then
    RUNS_TSV_ABS="$RUNS_TSV"
fi
if [ ! -f "$RUNS_TSV_ABS" ]; then
    echo "错误: 找不到 run 列表 TSV: $RUNS_TSV"
    echo "用法: sbatch training/cluster/run_compare_hemo_wss.slurm [相对仓库根的 tsv 路径]"
    exit 1
fi

HEMO_OUTPUT_DIR=${HEMO_OUTPUT_DIR:-"outputs/field/plots/optimization/wss_compare_${SLURM_JOB_ID}"}
HEMO_MAX_ITEMS=${HEMO_MAX_ITEMS:-0}
HEMO_NO_EXPORT=${HEMO_NO_EXPORT:-0}

export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=============================================="
echo "Field WSS compare (compare_hemo_wss_runs)"
echo "=============================================="
echo "作业ID: ${SLURM_JOB_ID:-local}"
echo "节点: ${SLURM_NODELIST:-$(hostname)}"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "Run 列表: $RUNS_TSV_ABS"
echo "输出目录: $PROJECT_DIR/$HEMO_OUTPUT_DIR"
echo "HEMO_MAX_ITEMS: ${HEMO_MAX_ITEMS} (0=全量)"
echo "HEMO_NO_EXPORT: $HEMO_NO_EXPORT"
echo "=============================================="

COMPARE_CMD=("$PYTHON_BIN" -u -m training.scripts.compare_hemo_wss_runs)

while IFS= read -r line || [ -n "$line" ]; do
    line="${line//$'\r'/}"
    if [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    if [[ -z "${line// }" ]]; then
        continue
    fi
    label="$(echo "$line" | cut -f1)"
    mpath="$(echo "$line" | cut -f2-)"
    label="${label//[[:space:]]/}"
    if [ -z "$label" ] || [ -z "$mpath" ]; then
        echo "跳过无效行: $line"
        continue
    fi
    mpath="${mpath#"${mpath%%[![:space:]]*}"}"
    mpath="${mpath%"${mpath##*[![:space:]]}"}"
    manifest_abs="$PROJECT_DIR/$mpath"
    if [ ! -f "$manifest_abs" ]; then
        echo "错误: manifest 不存在: $manifest_abs"
        exit 1
    fi
    COMPARE_CMD+=(--run "${label}:${manifest_abs}")
done < "$RUNS_TSV_ABS"

# 基底 4 项: python -u -m training.scripts.compare_hemo_wss_runs；每组 run 再 +2 (--run argv)
if [ "${#COMPARE_CMD[@]}" -lt 6 ]; then
    echo "错误: TSV 中未解析到任何 --run 行"
    exit 1
fi

if [ "$HEMO_NO_EXPORT" != "1" ]; then
    COMPARE_CMD+=(--export)
fi

if [ -n "$HEMO_MAX_ITEMS" ] && [ "${HEMO_MAX_ITEMS}" -gt 0 ] 2>/dev/null; then
    COMPARE_CMD+=(--max-items "$HEMO_MAX_ITEMS")
fi

COMPARE_CMD+=(--output-dir "$PROJECT_DIR/$HEMO_OUTPUT_DIR")

echo ""
echo "执行: ${COMPARE_CMD[*]}"
echo ""

"${COMPARE_CMD[@]}"
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "结束: $(date)  退出码: $EXIT_CODE"
echo "结果: $PROJECT_DIR/$HEMO_OUTPUT_DIR/wss_compare_summary.json"
echo "=============================================="

exit "$EXIT_CODE"
