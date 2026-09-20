# 沿程派生字段复核说明

`derived_audit.json` 对 170 例重算了截面坡度、上下游 10 mm 上下文和壁面插值字段，并用解析断点/缺失段自测验证了“不跨 invalid gap、排除当前站、并列取最近”的规则。

其中 `cases_with_errors` 不能直接读成 170 例算法失败：壁面字段的差异主要来自候选实现先应用 `wall_map_ambiguous` 再写 valid mask，而独立复核默认先做几何插值；这会产生成片的 mask disagreement。截面 slope/context 的数值复核在报告容差内。真正的验收结论以 `independent/geometry_compliance_audit.json` 与 `case_triage.csv` 的结构/语义分流为准。
