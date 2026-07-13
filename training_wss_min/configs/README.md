# training_wss_min/configs — 主题目录对照

配置按**实验主题**分目录，不再使用 `round5` / `round6` 等轮次目录名。

**兼容约定**：JSON 文件名可语义化；文件内 `name` 字段与历史 `runs/<name>/` **保持原样**（含 `r2_`/`r3_`/`r4_`/`r5_`/`r6_`），便于复评与追溯。新实验生成器应直接写语义 `name`。

| 目录 | 主题（核心问题） | 历史对照 | 状态 |
|---|---|---|---|
| `baseline_sweep/` | 点数 / 特征 / 采样正交扫 | 第 1 版 baseline | 归档 |
| `loss_aug_ablation/` | loss / 采样加权 / 旋转增强 | 原 `r2_*` | 归档 |
| `clean_data/` | clean-data 主矩阵（mse / tgtw × seed） | 原 `r3_*` | 归档 |
| `protocol_gates/` | 固定阈值、Gate-1、B/C 协议锚点 | 原 `r4_dev1_b*`/`c*` | 归档；锚点 `b1_tgtw_fixedq_*` |
| `pointcount_curve/` | 点数—精度曲线（xyz+geom） | 原 `r4_dev1_pc_*` | 归档 |
| `fit_lc_diagnosis/` | 拟合链 / LC / B-REP / A0* 诊断 | 原 `round5/` | 归档（科学结案） |
| `xyz_scale_diag/` | A/B/D XYZ 尺度诊断；C/E 后续 | 原 `round6/` | **A/B/D 已完成；C/E 为当前 P0** |
| `multitarget/` | 壁面压力与多目标横向诊断 | 第六轮 Track A/B | H-PW 已完成；Track B 按 Gate 触发 |
| `sweeps/` | 提交用 manifest（`*.txt`） | 原根目录 `sweep_*.txt` | 随主题更新 |

## 常用入口

```bash
# 已完成资产：壁面 gauge pressure 横向对比
WSSMIN_MANIFEST=training_wss_min/configs/sweeps/pressure_wall.txt \
  bash training_wss_min/cluster/submit_baseline_sweep.sh

# 协议锚点（B1 fixed target-weight）
training_wss_min/configs/protocol_gates/b1_tgtw_fixedq_s1234.json
```

生成器与汇总脚本见 `training_wss_min/tools/`（`-m training_wss_min.tools.…`）。实验指标与结论文档仍以 [`WSS最小化_训练实验跟踪.md`](../../docs/02-推进与变更/WSS最小化_训练实验跟踪.md) 为准。
