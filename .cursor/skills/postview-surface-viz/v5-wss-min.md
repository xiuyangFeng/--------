# V5 / WSS-min 可视化入口

仅在本次 run 属于 `training_wss_min/` 的壁面或体场路线时读取。路径、目标与 split 从 run 配置和 `training_wss_min/experiments/<batch>/` 核对；目录名不能证明它只读旧 `data_wss_min/`。

新交付统一遵循 [delivery-fields.md](delivery-fields.md)。下列工具是路线入口，已有工具的“文件齐全”不保证已含所需面片、selfmax 或正确索引，需按新合同补齐并验收。

## 直接 WSS 壁面包

`training_wss_min/tools/export_wss_postview.py` 含旧 WSS-min 与 V5 配准处理，但不等于支持每个新模型。已核对参数：`--run-dir`、互斥的 `--cases`/`--partition`、`--allow-test`、`--checkpoint best|last`、`--output-dir`、`--device`、`--resume`。

**2026-09-12 源码核查的兼容边界（使用时重查）**：导出器 `D.load_case` 没有传 `extra_point_features` / `point_features_root`，不能直接加载 C1 等依赖 curvature/tangent-normal sidecar 的新模型。已有正式 `predictions.npz` 时优先复用所选病例缓存，绑定 checkpoint、坐标及标量 schema，再打包；确需重推理时先完成特征加载和采样协议适配，不能直接套下面命令。

在确认模型兼容、导出已授权、目标资源已分配后，从仓库根运行；占位符替换为本次合同：

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python -m training_wss_min.tools.export_wss_postview \
  --run-dir '<run_dir>' --cases '<canonical_case_1>,<canonical_case_2>' \
  --checkpoint best --output-dir '<new_batch_dir>' --device '<allocated_device>'
```

- 已授权 test 病例加 `--allow-test`。CLI 开关不是新授权，也不能解除实验的 holdout 合同。
- 只有全分区导出任务才用 `--partition`，不能为省病例选择而改为全 test。
- `--resume` 用于补缺包；先核对已有 manifest 的 run、checkpoint、病例、指标与数据合同。脚本“文件齐全”检查不能证明父源全部一致。
- 帧号取 bundle 的 `peak_step`，V5 已有包常为 1162；旧 WSS-min 不强改为 V5 帧号。

| 产物 | 用途 |
| --- | --- |
| `_export/*__wall.csv` | 原始同点标量与误差；核对归一化/物理单位 |
| `*__pointcloud_wall.vtp` | 原点云对照；原量、训练归一化与 selfmax 分开命名 |
| `*__surface_wall.vtp` | STL + Gaussian `r=3 mm, sharpness=2, max_dist=3 mm` |
| `*__mapping_report.json` | 覆盖率、映射距离 |
| `manifest_bundle.json` | peak、partition、checkpoint、旋转、scale、normalization、pointcloud_metrics |
| `plots/fig_wss_triptych.png` | 历史工具默认使用训练 `true_norm/pred_norm/err_norm`；新交付另出原量和 selfmax 预览，不把该图改标题充数 |

`global_stats` 可按冻结训练统计恢复物理量；病例最大值归一化不能在无额外尺度时宣称恢复绝对 Pa。读 manifest 与标量键，不把 `true_norm/pred_norm` 指标写成 Pa 指标。

用户要求 Pa 云图时，先确认缓存/CSV 的物理恢复与正式评估一致，再用 `wss_cfd/wss_pred/err_wss` 绘制同色标图，单位 Pa；不能只修改默认 normalized PNG 的标题。

selfmax 从原始同点物理 WSS 的 CFD/Pred 各自最大值计算。缓存中的训练归一化字段保持原义；点云、wall-only CSV、native 和 Gaussian 均使用同一对固定分母，详见统一字段合同。

## V5 原生 CFD 壁面

母库有 `topology/wall_triangles`，节点身份与同点 CSV 一致时，默认另附直接挂载标量的 native 面片，便于检查 Gaussian 平滑。入口 `training_wss_min/tools/add_cfd_wall_mesh_vtp.py` 接受 `--batch-dir`、`--snapshot-root`，会修改现有病例与批次 manifest；运行前绑定本批母库并检查是否支持本批字段/布局。

输出 `*__cfd_wall_mesh.vtp` 应与 `*__surface_wall.vtp` 同帧。先核对顶点数/顺序和三角索引，再并排展示，注明“原生 CFD 壁面，无插值”与“STL Gaussian 回插”。

配准不能盲用历史 `R`/`R.T`：当前导出器通过 wall 坐标重建检查识别约定；manifest 含 `rotation_convention`、`stl_to_pipeline_scale`、`unit_extent_mismatch`。异常时查坐标/单位，不能用视觉缩放掩盖合同错误。

`legacy_vertex` 为顶点 hotspot 指标，`both_strict` 才请求严格面积映射。Gaussian coverage 即使 100% 也不能代替面积映射验收。

## V5 体场

`training_wss_min/tools/export_v5_volume_postview.py` 与 `plot_v5_volume_postview.py` 是已有 R5P/R5V 批次工具，内部固定 run/输出，导出器会重现全 test34；不是只换 `--run-dir` 的通用入口。仅在复现该批次且任务覆盖其运行范围时使用；其他模型先复用已验收缓存或显式参数化改造。

R5 历史合同：压力为 wall∪interior 的相对压力 `p-p_ref`（Pa），速度为 interior 的 `|u|`（m/s）。checkpoint、support 数/种子取该批冻结设置，不推广为其他模型默认值。

- `topology/wall_triangles` 可生成压力壁面，不能提供体四面体连接。
- 压力默认交付 wall∪interior 完整点云，以及 wall-only 源回插的 Gaussian 面片和可用的 native 壁面；不能仅给完整点云。先验证 `query_idx` 的拼接域，按 `concat(wall, interior)[query_idx]` 取坐标，避免壁面负索引挂到体内。
- 速度默认只保留 interior 点云及原 CFD/Pred 速度向量，不做壁面插值/投影。`point_kind` 仅按拼接位置生成，不能独立证明没有边界点：核对 `query_idx-n_wall` 对应的 volume 行及母库 `volume_static/cell_id_cas`、`source_row`、`xyz_mm`，并交叉核验同帧壁面三角面距离。现有 `volume_view.py` 的源约定是 anatomy cell centres，不能因近壁低速就改判成 wall。
- selfmax 使用速度模，压力 selfmax 使用完整原始评估域的相对压力，均遵循固定分母合同。完整点云发生遮挡时，可按 [几何查看合同](delivery-fields.md#体点身份与几何查看) 另附半剖/薄层点集；保留源索引、完整场和原分母，显示子集不改变正式指标或病例角色。
- slab/投影仅在任务需要时出图；真实连续截面需另取 Fluent 体单元并完成身份/坐标映射，点云 VTP 本身没有体单元。
- 速度向量随坐标同一旋转，相对压力保留参考零点；不能另行拟合偏置或预测幅值来改善图。

**旧 VTU 工具不能直接用于 V5 配准体场**：2026-09-12 的 `build_sliceable_volume.py` 只用 `--cas-scale` 缩放 raw `.cas`，没有 V5 刚性 R/translation 变换；会合并非空 `.cas` 块，没有 anatomy blood-zone 筛选，VTP 读取只接收单分量数组。连续 Slice 前需适配或预生成同帧、正确流域的体网格；速度向量场还需保留/旋转多分量数组。只看 speed 标量不需要向量箭头，但仍不能跳过注册与流域核验。

已有包仅用于查结构：`outputs/field/postview/wss_v5_r4_best_worst_20260907/`、`outputs/field/postview/wss_v5_r5_volume_best_worst_20260909/`。

新矩阵状态读 `docs/02-推进与变更/README.md` 指向的所属块实验跟踪（§0–§32 在 `00-V5设计与历史跟踪/` 历史卷）对应节。已完成的直接 WSS 批次入口为 `training_wss_min/experiments/wss_direct_recovery_20260912/README.md`，不在技能内复制排名或自动重跑。
