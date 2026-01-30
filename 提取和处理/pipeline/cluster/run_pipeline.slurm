#!/bin/bash
#SBATCH --job-name=gnn_pipeline          # 作业名称
#SBATCH --partition=compute              # 分区（根据集群配置修改）
#SBATCH --nodes=1                        # 节点数
#SBATCH --ntasks=1                       # 任务数
#SBATCH --cpus-per-task=8                # 每任务CPU核数
#SBATCH --mem=32G                        # 内存
#SBATCH --time=24:00:00                  # 最大运行时间
#SBATCH --output=logs/pipeline_%j.out    # 标准输出
#SBATCH --error=logs/pipeline_%j.err     # 错误输出

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
END_STEP=${3:-4}             # 结束步骤（默认4）
SAMPLING_METHOD=${4:-"fps"}  # 采样方法（默认fps）

# ============================================================================
# 环境配置 - 根据你的集群环境修改
# ============================================================================

# 加载模块（根据集群配置）
# module load anaconda3
# module load vtk

# 激活 conda 环境
source ~/.bashrc
conda activate rag_venv

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"

cd "$PIPELINE_DIR"

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
echo "工作目录: $PIPELINE_DIR"
echo "数据目录: $PROJECT_DIR/data_new"
echo ""
echo "参数:"
echo "  病例: ${CASE_NAME:-所有病例}"
echo "  步骤范围: $START_STEP → $END_STEP"
echo "  采样方法: $SAMPLING_METHOD"
echo "=============================================="
echo ""

# ============================================================================
# 运行 Pipeline
# ============================================================================

# 构建命令
CMD="python run_all.py"
CMD="$CMD --start-step $START_STEP"
CMD="$CMD --end-step $END_STEP"
CMD="$CMD --sampling-method $SAMPLING_METHOD"

if [ -n "$CASE_NAME" ]; then
    CMD="$CMD --case $CASE_NAME"
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
