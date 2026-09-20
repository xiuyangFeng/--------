# 启发式汇报 v2：网络模块补充（2026-09-13）

四页原生可编辑形状与连接线，16:9，字体 Noto Sans CJK SC；可复制到主 PPT。

- 第 1 页：R4 共用骨架，补充 LocalGeoPE、SA2 邻域注意力、PointNeXt-R 的具体作用。
- 第 2 页：X5 / X5X11 的 27D 输入、方向多尺度局部分支、完整壁面 K16 patch/FiLM 与截面 token 的注入点。
- 第 3 页：PF6 / VF6 独立体场模型的 20D 输入、BT 与局部分支差别、QAD-lite 残差解码。
- 第 4 页：X5D_v51。推理结构与 X5 相同；新增的是训练期密度增广（70/50/35/25%，p=0.6）。不是纵向几何 X5D_long。

## 文件

- [PPTX](启发式v2_网络模块补充_20260913.pptx)
- [PDF](启发式v2_网络模块补充_20260913.pdf)
- 单页预览：[第1页](slide-1.png)、[第2页](slide-2.png)、[第3页](slide-3.png)、[第4页 X5D](slide-4.png)
- [生成脚本](build_network_modules.py)

## 结构核验

冻结配置：`training_wss_min/configs/v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234.json`、`configs/wss_local_wave1_20260912/X5_s1234.json`、`configs/wss_local_wave2_20260912/{X5X11,PF6,VF6}_s1234.json`、`configs/wss_v51_wave1_20260916/X5D_v51_s1234.json`。前三者为相对 `training_wss_min/` 的路径。

实现：`training_wss_min/baseline_models.py` 的 `PointNetPlusPlusRegressor.encode_support/decode_query`、`LocalWallBranch`、`MultiRadiusSetAbstraction`；`training_wss_min/local_refinement.py` 的 `LocalPatchFiLMRefinement`、`SectionContext`。R4 对照既有 `docs/03-汇报材料/启发式实验汇报_2026-09-13/WSS_V5_R4_模型结构_20260909/README.md`。

必须保留的区别：

1. SA2 是局部注意力，PF6/VF6 的 BT 是粗层全局注意力。SA1/SA2 各一个 InvRes；固定中心数 125/125/32。
2. X5/X5X11 均有方向 support 局部分支与完整壁面 patch FiLM；query patch 默认 atlas 坐标与 mean pooling。方向分组不是 query patch 的切平面变体。
3. X5X11 的截面 token 由修正后的 support 特征与输入几何池化；投影为 256D 后加入病例粗层上下文，再拼接 query 32D 控制 FiLM。不是 SA3 BT。
4. VF6 已有普通 support 局部分支；SA3 内的 MultiRadiusSetAbstraction 实际仅用 `[0.20]` 一个半径，内部有 BT。PF6 用普通 SA3 后追加 BT，且无 support 局部分支。
5. PF6/VF6 分别训练，壁面 support 几何编码，体内位置为 query，不是共同输出 uvwp 的联合模型。
6. F6 是几何估计输入：`log_q_branch_murray` 与 `log_tau0_murray`，不是 CFD 真实分流输入或 PDE loss。模块功能描述不等于因果增益，数字证据应在主稿的配对消融页说明。
7. X5D_v51 的推理图与 X5 相同；密度增广只在训练期以 p=0.6 替换 70/50/35/25% 密度视图，测试/部署仍是全密度。第 4 页不是纵向几何 X5D_long，也不把 X11 画进默认底座。

已运行脚本生成与 slide 边界断言；LibreOffice 导出 PDF，pdftoppm 渲染全部四页。未启动训练或新增测试集评估。
