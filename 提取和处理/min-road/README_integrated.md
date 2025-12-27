# 整合清洗与合并脚本说明

## 📋 功能概述

`integrated_clean_merge_min.py` 是一个整合脚本，它将原先的两步流程（清洗 + 合并）整合为一步完成，**无需生成中间文件**，大幅节省磁盘空间。

### 🎯 核心特性

1. **最远点采样（FPS）**：确保血管几何结构的完整性，防止空间截断
2. **智能分层降采样**：优先保留近壁区高价值点
3. **动态预算补位**：最大化利用点数预算，不浪费
4. **内存处理**：无需生成中间文件，节省磁盘空间

### 原先流程的问题
```
原始数据 (ascii + ascii_in)  [大文件]
    ↓
清洗 → ascii_clean + ascii_in_clean  [占用大量空间]
    ↓
随机采样合并 → ascii_merged  [可能导致几何截断]
```

### 新流程的优势
```
原始数据 (ascii + ascii_in)  [大文件]
    ↓
内存清洗 + FPS分层采样 + 合并  [无中间文件，几何完整]
    ↓
ascii_merged  [最终结果]
    ↓ (可选)
替换 ascii_in 为降采样后的小文件  [节省空间]
```

### 📊 降采样策略

**采样逻辑（优化版）：**

1. **优先级 1（壁面）**：无条件保留所有壁面点
   - 计算剩余预算：`budget = target_total - len(wall_points)`

2. **优先级 2（内部点分层）**：使用 KDTree 计算距离
   - 近壁层：距离 < 2.0mm（高价值区域）
   - 核心层：距离 ≥ 2.0mm

3. **动态配额分配**：默认 7:3 比例
   - 近壁层：70% 预算
   - 核心层：30% 预算
   - **关键**：如果某层不足配额，自动转移给另一层

4. **采样方法（可选）**：

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **FPS（推荐）** | ✅ 空间分布均匀<br>✅ 防止几何截断<br>✅ 保持结构完整性 | ❌ 速度较慢（O(n×m)）<br>❌ 30-60秒/时间步 | 正式数据处理<br>高质量要求 |
| **Random** | ✅ 速度极快（O(n)）<br>✅ 5-10秒/时间步<br>✅ 简单高效 | ❌ 可能分布不均<br>❌ 有截断风险 | 快速测试<br>预览数据 |

**推荐策略**：
- 🔬 **研究/论文**：使用 FPS，确保数据质量
- 🚀 **快速迭代**：使用 Random，加速开发流程
- 💡 **混合使用**：Random 预览 → FPS 正式处理

## 🚀 快速开始

### 基本用法

```bash
# 处理单个病例（默认会替换 ascii_in 中的原始大文件）
python integrated_clean_merge_min.py "data/AAA/rupture/FENG_LI_XIN"

# 或者使用相对路径
cd min-road
python integrated_clean_merge_min.py "../data/AAA/rupture/FU_GUO_JUN"
```

### 高级选项

```bash
# 使用随机采样（速度快，适合快速测试）
python integrated_clean_merge_min.py "../data/AAA/rupture/FU_GUO_JUN" --sampling-method random

# 使用 FPS 采样（默认，质量高，适合正式处理）
python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" --sampling-method fps

# 不替换原始 ascii_in 文件（保留原始大文件）
python integrated_clean_merge_min.py "../data/fast/ZHANG_XIU_ZHEN" --no-replace

# 自定义分层比例（8:2，更倾向近壁层）
python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" \
    --boundary-ratio 0.8 \
    --core-ratio 0.2

# 自定义近壁区阈值和目标点数
python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" \
    --boundary-threshold 2.5 \
    --target-total 50000

# 组合使用：随机采样 + 不替换原始文件（快速预览）
python integrated_clean_merge_min.py "../data/fast/ZHANG_XIU_ZHEN" \
    --sampling-method random \
    --no-replace

# 设置随机种子以保证结果可复现
python integrated_clean_merge_min.py "../data/fast/ZHANG_XIU_ZHEN" --seed 42
```

## ⚙️ 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `case_dir` | 必需 | - | 病例目录路径（包含 ascii 和 ascii_in 子目录） |
| `--sampling-method` | 字符串 | fps | 采样方法：`fps`（最远点采样，质量高）或 `random`（随机采样，速度快） |
| `--no-replace` | 标志 | False | 添加此标志则不替换 ascii_in 中的原始文件 |
| `--boundary-threshold` | 浮点数 | 2.0 | 近壁区阈值（毫米） |
| `--boundary-ratio` | 浮点数 | 0.7 | 近壁层预算分配比例（0-1），默认 70% |
| `--core-ratio` | 浮点数 | 0.3 | 核心层预算分配比例（0-1），默认 30% |
| `--target-total` | 整数 | 40000 | 目标总点数 |
| `--seed` | 整数 | 1234 | 随机种子（用于 FPS 初始点选择或随机采样） |

**注意**：
- `boundary-ratio` + `core-ratio` 应等于 1.0（脚本会自动归一化）
- **采样方法选择**：
  - `fps`：推荐用于正式数据处理，质量高但速度较慢（~30-60秒/时间步）
  - `random`：推荐用于快速测试和预览，速度快但可能空间分布不均（~5-10秒/时间步）

## 📂 目录结构要求

脚本要求病例目录具有以下结构：

```
病例目录/
├── ascii/              # 壁面点原始数据
│   ├── XXX-0001
│   ├── XXX-0002.csv
│   └── ...
├── ascii_in/           # 内部点原始数据（会被替换为小文件）
│   ├── XXX-0001.csv
│   ├── XXX-0002.csv
│   └── ...
└── ascii_merged/       # 输出：合并后的数据（自动创建）
    ├── XXX-0001.csv
    ├── XXX-0002.csv
    └── ...
```

## 💾 空间节省示例

假设原始 `ascii_in` 文件：
- 原始大小：每个文件 100MB
- 10 个时间步文件 = 1000MB

处理后：
- 降采样后：每个文件约 5-10MB
- 10 个时间步文件 = 50-100MB
- **节省约 90% 空间**

## 📊 处理流程详解

### 1. 读取原始数据
- 从 `ascii` 读取壁面点
- 从 `ascii_in` 读取内部点

### 2. 内存清洗
- 统一列名映射
- 坐标单位转换（米 → 毫米）
- 补齐缺失列
- 添加 `is_wall` 标记

### 3. 智能分层降采样（核心优化）

**步骤 3.1：优先级 1 - 壁面点**
```
保留所有壁面点（无条件）
剩余预算 = 40000 - 壁面点数
```

**步骤 3.2：优先级 2 - 内部点分层**
```
使用 KDTree 计算每个内部点到壁面的最近距离
近壁层：距离 < 2.0mm
核心层：距离 ≥ 2.0mm
```

**步骤 3.3：动态配额分配**
```
初始配额：
  近壁层 = 剩余预算 × 70%
  核心层 = 剩余预算 × 30%

动态补位：
  if 近壁层实际点数 < 近壁层配额:
      多余配额 → 核心层
  if 核心层实际点数 < 核心层配额:
      多余配额 → 近壁层
```

**步骤 3.4：最远点采样（FPS）**
```python
# 伪代码
def farthest_point_sampling(points, n_samples):
    随机选择第一个点
    for i in range(1, n_samples):
        计算所有点到已选点集的最小距离
        选择距离最远的点
    return 采样点索引

近壁层采样 = FPS(近壁层点, 近壁层配额)
核心层采样 = FPS(核心层点, 核心层配额)
```

**FPS 优势**：
- ✅ 空间均匀分布
- ✅ 防止几何截断
- ✅ 保持血管结构完整性
- ❌ 随机采样可能导致某些区域缺失

### 4. 输出结果
- 保存合并数据到 `ascii_merged/`
- （可选）用降采样后的内部点替换 `ascii_in/` 中的原始文件

## ⚠️ 注意事项

1. **数据备份**：首次运行建议先备份 `ascii_in` 目录，或使用 `--no-replace` 选项

2. **文件命名**：脚本通过文件名中最后一个 `-` 后的数字匹配壁面点和内部点文件
   - 例如：`FENG_LI_XIN-0161.csv` 中的编号是 `0161`

3. **内存占用**：由于在内存中处理，处理大文件时会占用较多内存

4. **可恢复性**：如果需要恢复原始数据，建议：
   - 使用 `--no-replace` 选项，或
   - 提前备份原始数据

## 🔧 与旧脚本的对比

| 特性 | 旧流程（分两步） | 新流程（整合） |
|------|-----------------|---------------|
| 中间文件 | 需要 ascii_clean 和 ascii_in_clean | 无需中间文件 |
| 磁盘占用 | 约 3 倍原始数据大小 | 仅需最终结果空间 |
| 处理速度 | 需要两次读写 | 仅一次读写，更快 |
| 可选空间优化 | 无 | 可替换原始 ascii_in 文件 |

## 💡 推荐工作流程

### 方案 A：最大化节省空间（推荐）

```bash
# 1. 首次运行，替换原始文件
python integrated_clean_merge_min.py "data/AAA/rupture/FENG_LI_XIN"

# 结果：
# - ascii_merged/ 包含合并数据
# - ascii_in/ 被替换为小文件
# - 无中间文件
```

### 方案 B：保留原始数据

```bash
# 运行时不替换原始文件
python integrated_clean_merge_min.py "data/AAA/rupture/FENG_LI_XIN" --no-replace

# 之后如果确认结果正确，可手动删除 ascii_in 中的大文件
```

## 🎯 示例运行输出

```
📂 开始处理病例: FENG_LI_XIN
   近壁区阈值: 2.0mm
   预算分配比例: 近壁层 70% : 核心层 30%
   目标总点数: 40000
   采样方法: 最远点采样 (FPS)
   替换原始 ascii_in: 是

🔄 处理编号 0161...
   读取壁面数据: FENG LI XIN-0161
   读取内部数据: FENG_LI_XIN-0161.csv
   执行分层降采样合并（FPS）...
  壁面点数: 5432, 内部点数: 145678
  剩余预算: 34568 个点
  近壁层点数: 52341, 核心层点数: 93337
  最终分配: 近壁层 24197/52341, 核心层 10371/93337
  执行近壁层 FPS 采样...
  执行核心层 FPS 采样...
  ✅ 合并后总点数: 40000 (目标: 40000)
  预算利用率: 100.0%
   ✅ 合并文件已保存: FENG_LI_XIN-0161.csv
   🔄 已替换原始内部点文件: FENG_LI_XIN-0161.csv
      原始大小: 87.45MB -> 新大小: 2.15MB
      节省空间: 85.30MB (97.5%)

🎉 FENG_LI_XIN 处理完成!
```

**关键输出说明：**
- **剩余预算**：扣除壁面点后可用于内部点的配额
- **最终分配**：经过动态补位后的实际分配数量
- **FPS 采样**：使用最远点采样确保几何完整性
- **预算利用率**：实际使用点数 / 目标点数，应接近 100%

## 🔍 故障排除

### 问题：找不到匹配的文件

**原因**：壁面点和内部点文件的编号不匹配

**解决**：检查文件名格式，确保编号一致

### 问题：内存不足

**原因**：单个文件太大

**解决**：
1. 增加系统可用内存
2. 分批处理文件

### 问题：想恢复原始数据

**解决**：
1. 如果有备份，直接恢复
2. 如果没有备份，需要重新从 Fluent 导出

## 📝 开发者修改

如需在代码中直接使用，可以导入函数：

```python
from pathlib import Path
from integrated_clean_merge_min import process_single_case_integrated

# 示例 1：基本用法（FPS 采样，默认 7:3 比例）
process_single_case_integrated(
    case_dir=Path("data/AAA/rupture/FENG_LI_XIN"),
    replace_ascii_in=True,
    boundary_threshold=2.0,
    boundary_core_ratio=(0.7, 0.3),  # 近壁层 70%, 核心层 30%
    target_total=40000,
    sampling_method="fps",  # 使用 FPS 采样
    seed=1234,
)

# 示例 2：快速预览（随机采样）
process_single_case_integrated(
    case_dir=Path("data/AAA/rupture/FENG_LI_XIN"),
    replace_ascii_in=False,  # 不替换原始文件
    boundary_threshold=2.0,
    boundary_core_ratio=(0.7, 0.3),
    target_total=40000,
    sampling_method="random",  # 使用随机采样，速度快
    seed=1234,
)

# 示例 3：保守采样（8:2，更多近壁层点）
process_single_case_integrated(
    case_dir=Path("data/AAA/rupture/FENG_LI_XIN"),
    replace_ascii_in=True,
    boundary_threshold=2.0,
    boundary_core_ratio=(0.8, 0.2),  # 近壁层 80%, 核心层 20%
    target_total=50000,
    sampling_method="fps",
    seed=1234,
)

# 示例 4：激进采样（6:4，更多核心层点）
process_single_case_integrated(
    case_dir=Path("data/fast/ZHANG_XIU_ZHEN"),
    replace_ascii_in=False,  # 不替换原始文件
    boundary_threshold=1.5,
    boundary_core_ratio=(0.6, 0.4),
    target_total=30000,
    sampling_method="fps",
    seed=1234,
)
```

### 推荐配置

| 场景 | 近壁层:核心层 | 采样方法 | 说明 |
|------|--------------|----------|------|
| **默认（推荐）** | 7:3 | FPS | 平衡质量和效率 |
| **高精度壁面分析** | 8:2 | FPS | 更多近壁层点，适合 WSS 分析 |
| **全局流场分析** | 6:4 | FPS | 更多核心层点，适合速度场分析 |
| **快速预览** | 7:3 | Random | 快速查看数据质量 |
| **批量测试** | 5:5 | Random | 快速迭代参数 |
| **保守策略** | 9:1 | FPS | 极度重视近壁区 |

