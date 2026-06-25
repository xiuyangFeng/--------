# postview 参考：插值方法 · 产物结构 · QC

> 从 `docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md` 提炼；执行时以 **SKILL.md 默认参数 + 现有 shell** 为准。

## 方法选择

| 方法 | CLI | 用途 |
| --- | --- | --- |
| **gaussian** ★ | `--method gaussian --radius 3.0 --sharpness 2.0` | 汇报/ParaView 主图 |
| idw | `--method idw --k 8 --power 2` | 可解释平滑备选 |
| nearest | `--method nearest` | 坐标对齐 debug |

有 CFD 体网格（`.vtu`/`.cas+.dat`）时优先 `sample()`/Probe；**仅有 CSV 点云**时不假装网格插值。

### Gaussian 参数说明

- `radius` 太小 → 面片空洞；太大 → hotspot 被抹平
- `sharpness` 越大 → 远点权重越小
- `max_dist`：超半径标 NaN；报告 `valid_ratio`

### IDW 默认（若切换）

```text
k = 8 或 12
power = 2
max_dist = 2~5 mm
eps = 1e-8
```

---

## 病例包目录结构（单例）

```text
<CASE>__result_features_merged-1146/
├── _export/
│   ├── <stem>__wall.csv          # ★ 指标口径源
│   ├── <stem>__all.csv
│   └── <stem>__interior.csv
├── surface_gaussian/
│   ├── <stem>__stl_mapped_wall.vtp
│   ├── <stem>__stl_mapped_wall.csv
│   └── <stem>__mapping_report_wall.json
├── pointcloud/
│   └── <stem>__wall.vtp
├── plots/                          # CROWN 或额外三联图
│   ├── fig_p_triptych.png
│   ├── fig_wss_triptych.png
│   └── fig_vel_mag_triptych.png
├── <CASE>.stl
├── <CASE>__surface_wall.vtp        # ★ ParaView 主入口（副本）
├── <CASE>__mapping_report.json
├── GNN_blue_white_red.xml
├── README_后处理打开说明.md
└── manifest_bundle.json
```

### mapping_report.json 关键字段

```json
{
  "method": "gaussian",
  "params": { "radius": 3.0, "sharpness": 2.0 },
  "coverage": { "valid_ratio": 0.98 },
  "distance_mm": { "p50": 0.31, "p95": 1.12, "max": 4.83 },
  "metric_basis": "metrics are computed on original same-point CSV, not on interpolated STL"
}
```

---

## 展示帧对照

| 帧 | step | t_norm | 用途 |
| --- | ---: | ---: | --- |
| merged-1120 | 1120 | 0.00 | ❌ 压力梯度弱，单帧 R² 失真 |
| **merged-1146** | **1146** | **~0.16** | ✅ **默认主展示帧** |
| merged-1200 | 1200 | 0.50 | 可选对照 |
| merged-1280 | 1280 | 1.00 | 同 1120，周期端点 |

`t_norm = (step − 1120) / (1280 − 1120)`（`pipeline/convert_to_graph.py`）

---

## 变量与单位（VTP / CSV）

| 变量 | 含义 | 单位 |
| --- | --- | --- |
| wss_cfd / wss_pred | 壁面剪切应力 | Pa |
| p_cfd / p_pred | 压力 | Pa |
| vel_mag_cfd / vel_mag_pred | 速度模 | m/s |
| err_* | pred − cfd | 同左 |
| abs_err_* | \|err\| | 同左 |
| map_dist | 映射距离 | mm |
| map_valid | 有效插值 | 0/1 |

压力若做过 per-case offset correction，图注写 `per-case offset corrected`。

---

## QC 出图前清单

- [ ] STL 与 CSV 坐标单位一致（mm）
- [ ] 点云与 STL 无错位、旋转、镜像
- [ ] `valid_ratio` ≥ 95%
- [ ] `map_dist p95` 合理（相对血管局部半径）
- [ ] CFD 真值与预测 **同一插值法、同一参数**
- [ ] CFD/Pred 色标一致；误差图单独色标
- [ ] 图注：`method / radius / sharpness / coverage / 帧号 / t_norm`
- [ ] 主指标在原始 CSV / `summary.json`，非 VTP
- [ ] 分叉/薄壁处检查欧氏近邻是否跨面

---

## 图注模板

**WSS 三联图：**

> GUO_XI_JIANG · merged-1146 · t_norm≈0.16 · 收缩期上升段（近似）· Gaussian r=3 mm, sharpness=2, valid=98.9% · 指标为 81 帧 pooled R²_wss=0.429（非单帧）

**CROWN 压力：**

> CROWN non-PINN · ZHANG_JUN_HUA · merged-1146 · Gaussian r=3 mm · 壁面点云插值至 STL · 精度见 CROWN evaluate 表

---

## GNN vs CROWN 差异

| | GNN/V3P | CROWN |
| --- | --- | --- |
| 导出脚本 | `export_for_cfdpost.py` | `export_crown_for_cfdpost.py` |
| 打包脚本 | `package_postview_case.sh` | `package_crown_postview_case.sh` |
| 主三联变量 | `wss`, `p` | `p`, `vel_mag`（无 wss_pred） |
| 对照表 | `V3P_vs_CROWN_对照表_1146.md` | PINN 版 `V3P_vs_CROWN_PINN_对照表_1146.md` |

---

## 一句话口径（汇报用）

> 数值指标在原始同点点云 CSV 上计算；病例云图将 wss/p 用同一 Gaussian 核（r=3 mm）映射到同一 STL 顶点，并记录覆盖率与映射距离。
