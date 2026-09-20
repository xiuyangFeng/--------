# 历史 V3 / CROWN postview

仅用于历史 `outputs/field/` GNN/V3 与 CROWN。V5/WSS-min 使用 [v5-wss-min.md](v5-wss-min.md)，不能套用这里的帧号、pooled 图注和预测 manifest。

新建/补全的历史病例包也遵循 [delivery-fields.md](delivery-fields.md)：WSS/压力壁面回插并保留原点云，速度仅保留体内点云，新增原量与 selfmax 场。下列历史脚本和旧布局不代表已经满足新字段合同；按实际预测支持的变量适配，CROWN 不补造 WSS 预测。

## 输入与展示合同

- GNN/V3P：`outputs/field/<run>/predictions_test_best_wss/manifest.json` 或 `predictions_test/manifest.json`；用 `config.snapshot.json` 确认分支/checkpoint。
- CROWN：`external_baselines/crown_beihang/configs/local/` 配置与 `outputs/external_baselines/crown_beihang/<run>/best_model.pt`；本仓库该 baseline 没有 `wss_pred`，主变量为压力/速度，WSS 只能展示 CFD 真值。
- STL 从病例目录核对，不硬拼遗漏 slow/fast 的路径。
- 旧主展示帧为 `result_features_merged-1146`，`t_norm≈0.16`；1120/1280 是周期端点，按任务需要展示。
- 单帧云图引用 81 帧 pooled 指标时标注“全 81 帧 pooled，非此单帧”；未确认评估覆盖 81 帧时不能写 pooled。

## GNN ParaView 包

以下模板在仓库根、已授权导出与已分配资源下使用；环境先查脚本。常规导出/渲染为 GNN，Fluent 体网格读取可能需 GNN_vmtk。

```bash
MANIFEST='outputs/field/<run>/predictions_test_best_wss/manifest.json' \
CASE_NAME='<case_from_manifest>' SAMPLE_ID='result_features_merged-1146' \
RUN_TAG='<new_batch_tag>' \
bash tools/cfdpost_cloud_export/package_postview_case.sh
```

只要 WSS/P 三联图时，同样输入可用 `run_case_surface_compare.sh`；核对默认输出，避免覆盖旧批次。

完整包位于 `outputs/field/postview/<RUN_TAG>/<CASE>__<SAMPLE_ID>/`，含点云、STL 面片、mapping report、`manifest_bundle.json` 与打开说明；布局及插值细节见 [reference.md](reference.md)。

## CROWN 包

```bash
CROWN_CONFIG='<crown_config.json>' CROWN_CKPT='<run>/best_model.pt' \
CASE_NAME='<case_from_split>' SAMPLE_ID='result_features_merged-1146' \
RUN_TAG='<new_batch_tag>' CROWN_METHOD_LABEL='non-PINN' \
bash tools/cfdpost_cloud_export/package_crown_postview_case.sh
```

PINN 变体同时替换配置、checkpoint 和标签。`run_crown_surface_batch.sh`、`refresh_crown_surface_plots.sh` 含批次范围/默认路径，先检查，不用于替代只导出一个病例的请求。

## 已有 CSV，仅重映射

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python tools/cfdpost_cloud_export/map_to_stl_surface.py \
  --csv '<same_point_wall.csv>' --stl '<matching_surface.stl>' \
  --method gaussian --radius 3.0 --sharpness 2.0 --max-dist 3.0 \
  --scalars wss_cfd,wss_pred,p_cfd,p_pred \
  --output-dir '<new_out>/surface_gaussian'
```

只列 CSV 实际存在的标量。该命令是历史 raw 字段示例，不能据此省略新合同的 selfmax。正式交付时，CFD/Pred 原量同 stencil 回插后重算误差、绝对误差和固定分母 selfmax；不直接回插绝对误差或 valid mask。`nearest` 可诊断错位，汇报沿用 Gaussian 以保持可比。补 PNG 用 `plot_stl_mapped_triptych.py` 的 `--vtp`、`--render surface`、`--variable`、`--output`、`--report-json`；CFD/Pred 共用范围，误差单独对称色标，selfmax 另出预览。

## 体点云转连续截面

优先从 Fluent `.cas` 读取体单元，再映射体点云标量：

```bash
/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python tools/cfdpost_cloud_export/build_sliceable_volume.py \
  --cas '<source.cas.gz>' --source '<volume_pointcloud.vtp>' \
  --output '<new_out>/volume.vtu' \
  --cas-scale 1000 --radius 2.0 --sharpness 2.0 --fallback nearest
```

`--cas-scale 1000` 仅在已确认源为 m、目标为 mm 时使用；核对刚性注册，缩放本身不解决旋转/平移。检查非零单元、标量与源域，再在 ParaView Slice 中查看。

无 `.cas` 时 `--delaunay --alpha 4.0` 只是近似且 alpha 与单位相关；凹陷/分叉可能跨域补料，不能冒充原生 CFD 拓扑。

## 记录与方法

- 历史 V3 图件写 `docs/02-推进与变更/代码修改与实验推进记录.md`；CROWN 正式结果遵循 `docs/00-规范与记录/外部baseline实验记录规范.md`，不入内部母版表。
- 目录：`docs/00-规范与记录/点云回插面片可视化目录说明.md`。
- 方法：`docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md`。
- 工具：`tools/cfdpost_cloud_export/README.md`。
