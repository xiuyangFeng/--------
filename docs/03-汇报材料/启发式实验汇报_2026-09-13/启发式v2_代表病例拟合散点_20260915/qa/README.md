# QA

三张图均为 16×9 组会页，只导出 PNG。每张三个等宽 identity 散点，渲染时面板对齐 PASS。

按用户要求不再导出 PDF/SVG，也不再跑 PDF 文字/碰撞审计。静态 `validate_figure.py` 只扫 `plot_scatter_fit.py`，字体与 PNG 导出在 `plot_style.py`；KDE 的 6000 点抽样只用于着色，拟合与 R² 用全部同点。
