#!/bin/bash
#Author: XY
#SBATCH --partition=CPU                  # 分区
#SBATCH --nodes=1                        # 节点数
#SBATCH --ntasks-per-node=32             # 每节点任务数（192核可用，预留资源给系统）
#SBATCH --job-name=gnn_pipeline          # 作业名称
#SBATCH --output=logs/%x_%j.out          # 标准输出
#SBATCH --error=logs/%x_%j.err           # 错误输出
#SBATCH --time=04:00:00                  # 最大运行时间（单病例约1-2小时）
# 可选：指定节点
# #SBATCH -w node01

# ============================================================================
# GNN Pipeline - 集群运行脚本
# ============================================================================
# 使用方法:
#   sbatch run_pipeline.slurm                    # 处理所有病例
#   sbatch run_pipeline.slurm ZHANG_CHUN         # 处理单个病例
#   sbatch run_pipeline.slurm ZHANG_CHUN 2       # 从步骤2开始
# ============================================================================

# 获取命令行参数
CASE_NAME=${1:-""}           # 病例名称（可选）
START_STEP=${2:-1}           # 开始步骤（默认1）
END_STEP=${3:-5}             # 结束步骤（默认5）
SAMPLING_METHOD=${4:-"hybrid"}  # 采样方法（默认hybrid）
FPS_RATIO=${5:-"0.2"}        # hybrid 模式下 FPS 占比
ALLOW_NEAREST_BC=${ALLOW_NEAREST_BC:-0}
GEOMETRY_PYTHON=${GEOMETRY_PYTHON:-""}

# ============================================================================
# 环境配置 - 根据你的集群环境修改
# ============================================================================

# 清理并加载模块
module purge
# module load anaconda3
# module load vtk

# 激活 conda 环境
PIPELINE_ENV=${PIPELINE_ENV:-GNN}
GEOMETRY_ENV=${GEOMETRY_ENV:-GNN_vmtk}
DEFAULT_GEOMETRY_PYTHON="$HOME/.conda/envs/$GEOMETRY_ENV/bin/python"
if [ -z "$GEOMETRY_PYTHON" ] && [ -x "$DEFAULT_GEOMETRY_PYTHON" ]; then
    GEOMETRY_PYTHON="$DEFAULT_GEOMETRY_PYTHON"
fi

source ~/.bashrc
conda activate "$PIPELINE_ENV"

# 切换到提交目录
cd $SLURM_SUBMIT_DIR

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"

cd "$PROJECT_DIR"

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"

# ============================================================================
# 打印运行信息
# ============================================================================
echo "=============================================="
echo "GNN Pipeline - 集群运行"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "CPU核数: $SLURM_CPUS_PER_TASK"
echo "开始时间: $(date)"
echo ""
echo "工作目录: $PROJECT_DIR"
echo "数据目录: $PROJECT_DIR/data_new"
echo ""
echo "参数:"
echo "  病例: ${CASE_NAME:-所有病例}"
echo "  步骤范围: $START_STEP → $END_STEP"
echo "  采样方法: $SAMPLING_METHOD"
echo "  主环境: $PIPELINE_ENV"
echo "  几何环境: $GEOMETRY_ENV"
if [ -n "$GEOMETRY_PYTHON" ]; then
    echo "  geometry-python: $GEOMETRY_PYTHON"
fi
echo "=============================================="
echo ""

# ============================================================================
# 运行 Pipeline
# ============================================================================

# 构建命令
CMD="python -m pipeline.run_all"
CMD="$CMD --start-step $START_STEP"
CMD="$CMD --end-step $END_STEP"
CMD="$CMD --sampling-method $SAMPLING_METHOD"
CMD="$CMD --fps-ratio $FPS_RATIO"

if [ -n "$CASE_NAME" ]; then
    CMD="$CMD --case $CASE_NAME"
fi

if [ "$ALLOW_NEAREST_BC" = "1" ]; then
    CMD="$CMD --allow-nearest-bc"
fi
if [ -n "$GEOMETRY_PYTHON" ]; then
    CMD="$CMD --geometry-python $GEOMETRY_PYTHON"
fi

# 执行
echo "执行命令: $CMD"
echo ""

$CMD

EXIT_CODE=$?

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "=============================================="
echo "Pipeline 执行完成"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
