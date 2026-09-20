# 图件与交付核验（2026-09-15）

最终机器核验入口是 [final_verification.json](final_verification.json)，可用上一级 `verify_delivery.py` 复核。该报告记录源文件、图件、组装文件与脚本 SHA256，以及运行环境和检查范围。

## 最终结果

| 检查 | 结果 |
| --- | --- |
| 22 图 × PNG/PDF/SVG | 66 个导出文件齐全；PNG 均为 4800×2700 |
| 来源哈希 | 56 个不同来源文件 SHA256 与绘图 metadata 匹配 |
| PDF 文字 | 22 个图件均通过字体/字形审计；PDF 保留可选择文本 |
| PDF 碰撞 | 22 个图件全部 0 FAIL、0 WARN |
| 面板对齐 | 15 个多面板图 PASS，1.5 pt 公差；7 个单图信息流为 NOT APPLICABLE，不计作通过 |
| 组装文件 | 完整版 22 页、模块版 4 页，PPT/PDF 页数一致；逐页 PPT 图像字节和合订 PDF 文本与最终图件匹配 |
| 技能目录 | 20/20 目录存在，均有非空 SKILL.md 与资源文件；其中 19 个功能技能、1 个共享资源目录 |
| 绘图源码 | 四份依赖展开后的静态审计均无 FAIL；WARN 按下文逐项解释 |

PPT 使用整页高清图像加讲解备注，检查其图像内容和页数，不声称做过 PowerPoint 原生应用的逐页播放测试。图形修改入口为 SVG/PDF 与 Python 源码。

## 人工/逐图复核

模块图对照实际 `baseline_models.py`、`local_refinement.py` 与对应配置核对，逐图说明在 `03_backbone.review.json`、`04_x5.review.json`、`05_x11.review.json`、`06_volume_models.review.json`。量化图的逐面板记录见 [quantitative_panel_review.csv](quantitative_panel_review.csv) 与 [quantitative_review.json](quantitative_review.json)；病例页的 54 个面板记录见六份 `10/11/12*.visual_review.md`。

主流程查看过路线/条件/问题页、两页 loss、冲突诊断，复核当前病例分布、X11 模块和病例页/总览；病例绘制流程另逐图查看了全部六页。模块、量化、病例的来源检查分别进行；额外的只读复核确认安装完整性、X11 信息入口、旧 R5 与 VF6 区别、raw loss、Median 和 s1234 口径。这不是另一次完整独立模型评估。

病例面板保留原始几何和完整点数，不按误差剔除点；速度在本版绘图中也不抽样。单视角遮挡、不同病例物理色标不同、Gaussian 平滑的局限在图注/README 中保留。所有色标采用真实完整范围，没有通过截断高值放大视觉差异。局部误差观察仍需要截面或区域量化来支持机制解释。

## 静态源码 WARN 的处理

最终报告使用实际 `plot_style.py` 与实际绘图脚本拼接成的只读审计快照，弥补静态检查器不能追踪本地 import 的限制，不添加虚构绘图语句来满足检查。最终四份报告是：

- [build_modules_source_validation.json](build_modules_source_validation.json)：17 PASS / 4 WARN / 0 FAIL。
- [build_quantitative_source_validation.json](build_quantitative_source_validation.json)：18 PASS / 3 WARN / 0 FAIL。
- [build_case_plates_source_validation.json](build_case_plates_source_validation.json)：17 PASS / 4 WARN / 0 FAIL。
- [build_report_figures_source_validation.json](build_report_figures_source_validation.json)：17 PASS / 4 WARN / 0 FAIL。

共有三类全局格式提醒：没有 TIFF、300 dpi 未达默认 600 dpi、406.4 mm 画布不属于期刊 89/183 mm 栏宽。这里交付 16:9 组会材料和可编辑文本的 PDF/SVG，300 dpi PNG 用于 PPT，符合本次 [图件合同](../figure_contract.md)。不声称是某本期刊的最终生产文件。

模块/病例脚本的 `UNCERTAINTY-ENCODING` 提醒由 seed 文字触发。模块图说明结构用途，涉及增益时另列三个 seed 的配对值并在量化页展示；病例图来自一个固定 s1234 checkpoint，三 seed 只用于选例，不应给单个场附上虚构误差带。当前分布页已经显示样本 SD。

总体汇报脚本的 `LOG-GUARD` 提醒属于模式识别限制：每项 log 曲线在绘制前实际断言 `np.isfinite(y).all() and np.all(y > 0)`，不使用伪计数，不悄悄删除非正值。全部 9,985 个 epoch 均保留。

`_intermediate/` 与文件名含 `import-unaware` 的报告仅记录初次审计；旧模块/量化审计保留作过程追溯。最终状态以上述四份统一报告与 `final_verification.json` 为准。

## 科学结论边界

同 seed 配对支持方向一致，未做显著性推断；X11 的 MAE 和 IoU 代价同时展示。所有病例分布保留全部 34/35 例及负 R²。病例 R² 均值、R²_cb、三 seed 选例分数、s1234 原物理量分数分别标注。selfmax 仅用于分布对照，不重算或替换正式 R²。

V4 曲线与梯度是历史单 seed 诊断，未收敛状态和数据/协议差别已披露；旧 R5V 的 CFD 速度参照含冻结校准器和旧坐标尺度问题。未新增 VF6→WSS 闭环结果，也未声称某个速度/物理错误来源已经被唯一定位。此次未训练、推理或重新评估全测试集。
