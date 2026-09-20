# 10_x5x11_cases_selfmax：终版逐面板验收

已查看各页整图、病例单格及终版六页总览；本记录只说明展示与来源核验，不增加实验结论。

- 9 面板对齐：PASS；1.5 pt 公差；色条与旁注通过显式 axes 列表排除。
- 最终 PDF 碰撞：PASS，0 FAIL / 0 WARN。
- 最小实际字形：9.0 pt；低于 5 pt：0。
- 面板字母均完整处于浅灰图框内部；9 个 contained fill overlays 是字母位于背景中的正常包含，无遮蔽数据。
- CFD/Pred 同列相同视角、相同几何范围与色标；error 使用完整源值的对称范围。
- selfmax 严格使用源 manifest 中的完整同点评估域最大值，没有按面图或可见点重算分母。
- 全量源节点/点保留，排除数为 0；VF6 绘制全部体内点，没有三角面或壁面插值。
- 物理 R² 来自 s1234 固定 checkpoint；三 seed 均值仅选择病例，图中不报告归一化 R²。
- 保留边界：单个正交视角会遮挡背面/深处，空间现象需区域量化或截面验证；跨病例物理色标不同。

| 面板 | 病例角色 | 证据角色 | 源点数 | 表示/区间 | 同列编码与完整几何 | 结果 |
|---|---|---|---:|---|---|---|
| a | best | reference anatomy/field | 14,399 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| b | median | reference anatomy/field | 9,704 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| c | worst | reference anatomy/field | 14,958 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| d | best | same-point prediction | 14,399 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| e | median | same-point prediction | 9,704 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| f | worst | same-point prediction | 14,958 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| g | best | local discrepancy/failure boundary | 14,399 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| h | median | local discrepancy/failure boundary | 9,704 | 固定同点场；无重复试验区间 | 已核对 | PASS |
| i | worst | local discrepancy/failure boundary | 14,958 | 固定同点场；无重复试验区间 | 已核对 | PASS |

静态源码审计由主流程对实际绘图脚本及共享样式依赖展开后执行；单文件扫描不能推断导入模块中的字体和导出设置。
