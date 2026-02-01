#!/bin/bash
#Author: XY
# ============================================================================
# 生成病例列表文件，供 Array Job 使用
# ============================================================================
# 使用方法:
#   ./generate_case_list.sh              # 扫描所有病例
#   ./generate_case_list.sh CASE1 CASE2  # 指定病例
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_LIST_FILE="$SCRIPT_DIR/case_list.txt"

# 如果指定了病例
if [ $# -gt 0 ]; then
    echo "使用指定的病例列表..."
    > "$CASE_LIST_FILE"
    for case in "$@"; do
        echo "$case" >> "$CASE_LIST_FILE"
    done
else
    # 自动扫描数据目录
    DATA_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/data_new/AG/fast"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "错误: 数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    echo "扫描数据目录: $DATA_DIR"
    ls -d "$DATA_DIR"/*/ 2>/dev/null | xargs -n1 basename > "$CASE_LIST_FILE"
fi

# 统计并显示
CASE_COUNT=$(wc -l < "$CASE_LIST_FILE" | tr -d ' ')

echo ""
echo "=============================================="
echo "病例列表已生成: $CASE_LIST_FILE"
echo "=============================================="
echo "病例总数: $CASE_COUNT"
echo ""
echo "病例列表:"
cat -n "$CASE_LIST_FILE"
echo ""
echo "=============================================="
echo "下一步操作:"
echo "=============================================="
echo "1. 编辑 run_array.slurm，修改 --array 参数:"
echo "   #SBATCH --array=0-$((CASE_COUNT-1))%6"
echo ""
echo "2. 提交 Array Job:"
echo "   sbatch run_array.slurm"
echo ""
echo "或者使用快捷命令提交（自动设置 array 范围）:"
echo "   sbatch --array=0-$((CASE_COUNT-1))%6 run_array.slurm"
echo "=============================================="
