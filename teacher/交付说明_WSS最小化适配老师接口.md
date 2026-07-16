# WSS 最小化：按老师源码风格适配的交付说明

> 日期：2026-07-14
> 原则：**外形跟着老师走，内容换成我们项目。**
> 老师给的是写法模板；我们交回同结构的两份脚本，任务改为本项目 WSS-min。

---

## 1. 交付原则

| | 老师原版 | 我们交的适配版 |
| --- | --- | --- |
| 建库 | `sampling_buid_dataset.py` | `build_wss_dataset.py` |
| 训练 | `model_train_1.py` | `model_train_wss.py` |

**风格对齐：**

- 建库：`read_data` → `process_case` → `process_directory` → `split_and_save_dataset`
- 训练：`get_model` / `get_loss` / `train_step` / `train` / `my_collate`
- 训练循环：batch 内采点 → AMP → 按 train loss 存 `weight/best_epoch.pth`
- 已删掉老师版中未使用的 `pv_phy` 物理场占位

**内容换成我们项目：**

- 建库：**读已预处理的 `bundle.npz`**（不在此脚本里做 ascii/ascii_in）
- 输入：`xyz + abscissa_norm + local_radius + curvature`（6 维）
- 标签：峰值壁面 **WSS 标量**（1 维）
- 划分：AG 正式 train/val/test
- 训练侧：WSS `log_z`；几何 signed-log1p + z-score；FPS 2000

原始 Fluent → bundle 的预处理仍由工程仓 `pipeline_wss_min` 完成；本交付建库脚本只负责「读 bundle → 打 pkl」。

---

## 2. 样本格式

```text
(case_id, features(6, N), wss(1, N))
```

对应老师：`(basename, points(3, N), values(4, N))`

---

## 3. 怎么跑

```bash
# 建库（读 data_wss_min/AG/*/bundle.npz；建议 GNN）
cd teacher
python build_wss_dataset.py

# 训练
python model_train_wss.py
```

产物：`outputs/wss_pkl/wss_{train,val,test}.pkl`、`wss_norm_stats.json`；`weight/best_epoch.pth`。

---

## 4. 相对老师原版

| 老师 | 我们 |
| --- | --- |
| 读 wall CSV | 读 `bundle.npz` |
| 随机 split | 正式 train/val/test |
| `(3,N)+(4,N)` | `(6,N)+(1,N)` |
| MLP `3→4` | MLP `6→1` |
| randperm | FPS 2000 |
| p min-max | WSS log_z + 几何 z-score |

---

## 5. 目录

```text
teacher/
├── sampling_buid_dataset.py / model_train_1.py   # 老师原版
├── build_wss_dataset.py                          # 读 bundle → pkl
├── model_train_wss.py                            # 读 pkl → 训 WSS
└── 交付说明_WSS最小化适配老师接口.md
```
