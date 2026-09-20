# 后处理病例包：交付与字段合同

新建或补全病例包时读取。按本次变量、病例和时间范围生成，不把历史某批的模型、9 例或 1162 帧设为未来默认。用户只要求解释现有图或修改单张图时，不额外扩张为完整导出。

## 按变量交付

| 变量及原始评估域 | 默认可打开的场文件 | 附属材料 |
| --- | --- | --- |
| WSS：wall | 原始壁面点云 VTP；Gaussian STL 面片 VTP；有对应 CFD wall topology 时附未平滑 native 面片 VTP | 配准后几何、wall-only 插值源 CSV、回插顶点 CSV、mapping report |
| 速度：interior | 原始体内点云 VTP，保留速度大小、CFD/Pred 三分量向量及向量误差 | 不默认做壁面回插或近壁投影；原始向量与坐标同帧 |
| 压力：wall∪interior | 完整点云 VTP；仅以 wall 点为源的 Gaussian STL 面片；有对应拓扑时附 native 面片 | 保留点类型与相对压力参考；配准几何、wall-only 插值源 CSV、回插顶点 CSV、mapping report |

目标若只有壁面或只有体内，按真实源域交付并写明；缺少壁面预测时不能用内部压力或零速度伪造壁面预测。历史 CROWN 无 WSS 预测时只保留实际存在的 CFD WSS，不能制造 `wss_pred` 或其 selfmax。

每例另附完整原始同点字段 CSV（可 gzip）、manifest、打开说明和原量/selfmax 预览；整批附 README、打开文件清单、预览总览与核验报告。文件名可沿用有效的路线约定，以下名字仅为清晰的布局示例：

```text
<batch>/
  README.md
  打开文件清单.csv
  manifest.json
  verification.json
  plots/
  <model>/<role-or-case>/
    <case>__pointcloud.vtp
    surface_gaussian.vtp              # WSS / 有壁面预测的压力
    cfd_wall_native.vtp               # 同身份 CFD 壁面拓扑可用时
    aligned_geometry.vtp              # 用于面片配准复查
    _export/same_point_fields.csv.gz
    _export/gaussian_source.csv.gz    # wall-only；固定分母仍来自完整原始域
    surface_gaussian/mapped_vertices.csv
    mapping_report.json
    manifest.json
    README_打开说明.md
```

WSS/压力表面预览必须渲染实际三角面；速度预览用原体点云，降采样仅用于绘图且标注，不影响原场文件与分母。CFD/Pred 同一相机、同一显示范围，有符号误差用零点居中的对称色标。原量和 selfmax 分别出图，注明后者无量纲、已移除幅值差。

输出 VTP 必须自包含；包内 README、manifest `files`、打开清单与预览链接使用相对路径。绝对源路径仅用于追溯，并配来源哈希；不要让打开场文件依赖原工作区 symlink。脚本根据自身位置或参数定位输出，不能写死用户已经移动的旧路径。

## selfmax 字段与固定分母

用户所需的是分别除以各自最大值，不是共用 CFD 分母。先从本病例、本次所选帧的完整原始同点评估数组计算一次：

```text
D_cfd  = max(raw_cfd)
D_pred = max(raw_pred)
cfd_selfmax  = raw_cfd  / D_cfd
pred_selfmax = raw_pred / D_pred
selfmax_error_pred_minus_cfd = pred_selfmax - cfd_selfmax
selfmax_abs_error = abs(selfmax_error_pred_minus_cfd)
```

| `prefix` | 分母来源 / `denominator_scope` | 保留的物理含义 |
| --- | --- | --- |
| `wss` | 原始同点评估 wall 的 WSS 标量 | 原量 Pa |
| `speed` | 原始同点评估 interior 的 `norm(velocity, axis=1)` | 原速度大小和三分量向量 m/s；不能用分量各自最大值代替速度模 |
| `pressure` | 原始同点评估 wall∪interior 的压力标量 | 原量 Pa；若为 `p-p_ref`，原有参考零点保持不变 |

“完整”指本次缓存/正式评估实际定义的同点域；记录点数及采样合同，不补造未评估点。多帧分别明确病例/帧分母，不能默认为整个周期最大值。

每个可预测标量输出以下新字段，原有单位字段与原量误差保持不变：

| 字段 | 类型 / 含义 |
| --- | --- |
| `{prefix}_cfd_selfmax` | CFD / 本病例本帧 CFD 最大值，无量纲 |
| `{prefix}_pred_selfmax` | Pred / 本病例本帧 Pred 最大值，无量纲 |
| `{prefix}_selfmax_error_pred_minus_cfd` | 两个 selfmax 相减，无量纲 |
| `{prefix}_selfmax_abs_error` | 上述误差绝对值，无量纲 |
| `{prefix}_cfd_selfmax_valid` | CFD 分母和当前值有效，0/1 |
| `{prefix}_pred_selfmax_valid` | Pred 分母和当前值有效，0/1 |
| `{prefix}_selfmax_error_valid` | 上述两者均有效，0/1 |

历史 raw 字段如 `p_cfd`、`vel_mag_cfd` 可保留，适配到统一 `pressure`、`speed` selfmax 前缀，在 manifest 明确原字段映射。训练 `true_norm/pred_norm` 或 `wss_*_norm` 单独标为训练归一化；即使数值偶然相同也不能换名当作 selfmax。

分母在同一病例/帧的完整点云、wall-only 源 CSV、native 面片和 Gaussian 面片间固定复用。压力面片的分母仍取完整 wall∪interior 域，不在 wall 子集重算。Gaussian 先用相同权重插值原量 CFD/Pred，再分别除以这对原始分母并重算误差；不能在平滑后重求最大值、插值绝对误差或平均 valid mask。面片的 selfmax 最大值小于 1 是可能且正确的，不要再次缩放。

压力严格使用 `max`，不能替换为 `max(abs(...))`、min-max、截断或拟合偏置。相对压力 selfmax 可以为负且小于 −1；若整个场为负，除以其负最大值也可能大于 1，记录分母符号并按实际联合范围出图。不要把压力 selfmax 固定裁成 [0,1]。selfmax 只比较相对于各自最大值的分布，不能作为原量预测精度提升的证据；正式 R²、MAE、原量误差始终保留且来自原始同点域。

分母为非有限数或绝对值接近 0 时，该侧 selfmax 记 NaN、valid=0；相应误差也记 NaN，原量保留。阈值与原量单位一起写入 contract，不能以 1 替换分母、隐式使用 `nanmax` 或把无效值强行变为 0。插值覆盖无效的点同样 NaN+mask，区分 denominator invalid 与 mapping invalid。

可复用实现：`tools/cfdpost_cloud_export/display_fields.py`。使用前核对当前函数及返回 schema：

```python
contract = fit_selfmax(cfd, pred, prefix=prefix,
                       denominator_scope=scope, source_unit=unit)
arrays.update(selfmax_fields(cfd, pred, contract))
# 对映射/子域值复用同一个 contract，不再次 fit_selfmax。
surface_arrays.update(selfmax_fields(mapped_cfd, mapped_pred, contract))
field_data = normalization_field_data(contract)
```

manifest 保存 contract 的 `kind=selfmax`、`cfd_max`、`pred_max`、`denominator_scope`、来源单位/字段、病例/帧和源点数、formula、零值阈值/有效状态及 `invalid_policy`。VTP FieldData 写入对应分母和合同；常量不必复制成数十万逐点字段。原始同点 CSV 和映射 CSV 均包含适用的新字段和 masks，CSV 对应 manifest 中的同一合同。

## 坐标与回插身份

- 优先复用已验收预测缓存。检查 checkpoint/config/source hashes、选例指标来源、出图种子和原始同点指标；本地重打包不触发新训练或整队列推理。
- V5 体场 `query_idx` 若定义在 `concat(wall, interior)`，坐标必须从该拼接数组索引，并核验 `point_kind == (query_idx >= n_wall)`。不能对所有点一律 `query_idx-n_wall` 后索引 volume，负索引会把壁面值挂到体内；也不能用会提前计算两侧的 `np.where` 掩盖越界。以当前缓存合同为准，不推广到别的 schema。
- 按 node ID / row index / query index 同时核对坐标与真值来源；PF 压力壁面用本次体场合同对应的相对压力，而非母库里不同零点的绝对压力。只验点数无法发现错位。
- 由原点与对齐点重建检验旋转约定：例如 `(raw_mm-centroid) @ rotation.T` 必须数值通过后才用。STL 单位与节点对应关系独立核对，不用 bbox 拟合掩盖尺度或注册错误；速度向量同步旋转。
- native 面片须验证顶点身份、顺序、三角索引与有效点裁剪映射；不能只因顶点数相等就挂值。Gaussian 壁面源严格限制为该变量壁面点，并与 native 同帧。
- Gaussian 优先沿用已核验路线参数；当前常用 `radius=3 mm, sharpness=2, max_dist=3 mm, fallback=mask` 不是免检结论。报告实际参数、覆盖率、距离分位数、有效面数及薄壁/分叉混合诊断。native 用于检查平滑是否抹掉热点；覆盖率 100% 不等于无跨壁混合或面积指标通过。

## 体点身份与几何查看

- `point_kind=1` 可能仅表示 query 来自拼接数组的 volume 段，不是独立的边界判据。沿 `query_idx → volume 行 → 母库 cell/source identity` 核对源坐标与场值，确认是体单元中心还是边界节点；身份、顺序和坐标须同时对应。
- 以同帧真实 CFD 壁面三角面计算或交叉核验体点距离，记录单位、数值精度及合理容差。最近壁面**节点**距离只能辅助检查，不能代替到壁面**面片**的距离。近壁单元中心即使速度很低、距离很小仍可能是真实体点；零/低速度及截图中的外层遮挡都不能单独作为删点依据。
- 若确有混入的边界点，按源拓扑/身份与几何证据剔除，并保存原索引、精确规则和前后点数；显示子域与正式评估域分开标注。不能借此覆盖正式 R²、重排已选病例或悄悄改变 selfmax 分母。
- 完整体点云遮挡内部时，可另附几何 halfcut（半空间）或 slab（薄层）点集。用坐标定义平面原点、单位法向及薄层厚度，记录坐标帧、具体布尔谓词、保留索引和前后点数；CFD/Pred/Error 共用同一掩码。参数依病例和查看目的确定，不按预测误差或速度阈值挑点。
- 完整原点云保持独立可打开；辅助文件明确标为几何查看子集，薄层点云不是有体单元的连续 Slice。子集复用完整原始评估域的 selfmax 分母；`source_point_count` 指分母域点数，可不同于 `exported_point_count` / `displayed_point_count`。正式 R²保留原域标签；如另算子域指标，单列其范围，不能替换正式值。

## 交付前核验

1. 对每份新写出的 VTP 重新读取：完整点云保持源点数/身份/坐标；几何查看子集按记录的原索引核对点数与字段。WSS/压力面片必须有非零三角面；速度点云保留三分量数组、不产生壁面投影，并按上述身份与面片距离核验其真实域。
2. 从缓存和母库核验原量、压力参考与原始 R²/MAE，在数值容差内与正式记录一致。选例的多种子统计和出图单次指标不得混写。
3. 从原始完整同点域独立重求两个最大值，并核对 manifest 与所有 VTP FieldData 一致。逐点核验 selfmax 公式、signed/absolute error 和 0/1 mask；无效分母遵循 NaN 规则。不能只检查字段存在。
4. Gaussian 面片核对原量 Pred−CFD、固定分母 selfmax 及误差；核查 mapping mask、覆盖率、距离与配准，不在插值域重算正式精度。CSV 与对应 VTP/源数组在记录的精度下匹配。
5. 打开实际生成的原量/selfmax 代表预览，检查相机、图例、单位、负压力范围、色标和裁剪。记录预览是否绘图抽样；字段验收必须使用完整产物。
6. 检查所有包内相对链接和打开清单可解析，旧缺字段/错位产物移至明确标识的归档且不再列为主入口。验收报告写出检查依据、容差、结果与实际未满足项；不能仅写“生成成功”。
