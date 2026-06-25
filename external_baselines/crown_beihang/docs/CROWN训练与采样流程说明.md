# CROWN/Beihang 训练与采样流程说明

> 代码入口：`external_baselines/crown_beihang/train.py` · `evaluate.py` · `data.py`  
> 配置：`configs/local/crown_original_vp_split_AG_v1_seed1.json`（非 PINN 5751）· `crown_original_vp_pinn_split_AG_v1_seed1.json`（PINN 5757）

---

## 1. 数据层级：病例 → 帧 → 点

| 层级 | 含义 | AG split_AG_v1 规模 |
| --- | --- | --- |
| **病例（case）** | 一位患者的 AG 域 CFD 序列 | train 57 · val 8 · test 16 · 共 81 活跃例 |
| **帧（frame / sample）** | **一个 CFD 时间步** 的体素化点云；Dataset 中 **一条样本** | train **4617** · val **648** · test **1296** · 共 **6561** 帧 |
| **点（point）** | 0.05 mm 体素化后的空间点；每点特征 `x,y,z`，标签 `u,v,w,p` | **57.5 万–130.4 万/帧**，均值 **≈81.5 万** |

**「分帧」是什么？**

- 在本项目中，**帧 = 心动周期内的一个瞬态 CFD 解**，不是视频意义上的「分镜」，也不是把一帧再切成空间块。
- 每个病例约 **81 帧**：Fluent 输出步号 **1120–1280**（步长 2），对应一个完整心动周期；`t_norm = (step − 1120) / 160`。
- 预处理时 `export_pkl.py` 对 **每一帧** 单独体素化，写入 partial pkl；索引键为 `sample_id = "{case_name}/{frame_key}"`（例如 `GUO/merged-1146`）。
- `CrownLazyDataset.__getitem__` 每次返回 **一整帧的全点云**（数十万到百万级）；**不会在 Dataset 层做 1 万点采样**。

### 1.1 Fluent 脉动 CFD 与心动周期（AG 数据）

本节说明 **AG 私有数据** 的时间轴与周期内相位；与 Liao 2025 原文 Nagahama **稳态 CFD** 不同。

#### 仿真与截取策略

| 项 | 数值 | 说明 |
| --- | --- | --- |
| 求解器 | ANSYS Fluent **非定常**（脉动）N-S | 入口为时变体积流量 `vf-in`（`Global_conditions/vf-in-rfile.out`） |
| 物理时间步 | **Δt = 0.005 s** | 与 monitor 文件 `flow-time` 列一致（步号 × 0.005 s） |
| 总计算步数 | **1280 步** | 总物理时长 **6.4 s**（≈ 8 个心动周期） |
| 保留区间 | 步号 **1120–1280** | 物理时间 **5.6–6.4 s**，时长 **0.8 s** |
| 导出步长 | **偶数步**（1120, 1122, …, 1280） | 共 **81 帧**；相邻保留帧间隔 **0.01 s** |
| 隐含心率 | **75 bpm** | 0.8 s/周期 → 60/0.8 = 75 次/分（静息心率常见范围） |
| 为何只留最后一周期 | 前 ~5.6 s 为启动瞬态 | 多周期后流场进入 **周期稳态**（limit cycle），末周期与前一周期重复性最好 |

前 1119 步及奇数步在 `fluent.slurm` 后处理中删除，仅保留末周期偶数步 ASCII / `ascii_in`。

#### 周期内时间坐标

```text
t_norm = (Fluent_step − 1120) / (1280 − 1120)   ∈ [0, 1]
t_phys = step × 0.005 s                         （相对仿真起点）
```

**1120 与 1280 为同一周期首尾**（`t_norm=0` 与 `1` 相接）；不是「1120=舒张、1280=收缩」的简单二分。

#### 生理学分期（血流动力学近似，非 ECG 标定）

心动周期分为 **收缩期（systole）** 与 **舒张期（diastole）**。在 75 bpm、周期 0.8 s 时，收缩期约 **0.25–0.30 s（~30–38%）**，舒张期约 **0.50–0.55 s（~62–70%）**——与脑血管 CFD 入口流量波形一致（见下表）。

下表以 `CHEN_SHI_MING`（`fast`）入口体积流量为参考；`GUO_XI_JIANG`（`slow`）峰值/最低点 **步号相同**（`AG/fast` vs `AG/slow` 是病例血流分类，**不是**心率快慢）。

| `t_norm` | Fluent step | 帧序号 | 入口流量（相对峰值） | 血流动力学相位（近似） | 场特征 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.00 | 1120 | 1/81 | ~14% | **舒张末期**（end diastole） | 流量低；壁面压力空间梯度弱 |
| 0.06–0.12 | 1132–1136 | — | ~11% | **等容收缩早期**（isovolumetric contraction） | 流量仍低，心室压力上升、主动脉瓣尚未充分开放 |
| **0.16** | **1146** | **14/81** | **~41%** | **收缩期上升段 / 射血早期**（early ejection） | 流量快速上升；**V3P/CROWN 汇报主展示帧** |
| 0.25–0.26 | 1160–1162 | — | **~99–100%** | **收缩期峰值 / 最大射血**（peak systole） | 周期内 **最大流量**（本例 step 1162） |
| 0.33–0.40 | 1172–1184 | — | ~85–50% | **射血后期 / 收缩末期**（late systole） | 流量下降，惯性仍维持较高 WSS |
| 0.50 | 1200 | 41/81 | ~22% | **舒张早期**（early diastole） | 快速充盈段，流量继续降低 |
| 0.55–0.68 | 1210–1226 | — | &lt;10% | **舒张静止期**（diastasis） | 流量近零（本例最低 step **1216**，`t_norm≈0.60`） |
| 0.75–1.00 | 1240–1280 | — | ~14% | **舒张晚期**（late diastole） | 流量仍低，临近下一周期收缩 |

```text
入口流量 Q(t) 示意（相对峰值，CHEN_SHI_MING）:

Q
1.0 |        ****
    |      **    **
0.5 |    **        **
    |  *              *
0.0 |_*_________________*___  t_norm
    0   0.16  0.26      0.6   1.0
        ↑     ↑         ↑
      1146  峰值射血  舒张静止
```

#### 与临床概念的对照（汇报时可引用）

| 临床/生理学概念 | 在本项目 CFD 中的对应 |
| --- | --- |
| **收缩期** | `t_norm` 约 **0.05–0.40**（流量自低值升至峰值再回落） |
| **舒张期** | `t_norm` 约 **0.40–1.00**（流量低至近零再缓慢回升） |
| **等容收缩** | 流量尚未明显上升（`t_norm` &lt; ~0.12） |
| **心室射血** | 流量上升至峰值（`t_norm` ~0.12–0.30） |
| **等容舒张** | 流量自峰值快速下降（`t_norm` ~0.30–0.45） |
| **快速充盈 / 舒张静止** | 流量低谷（`t_norm` ~0.45–0.70） |
| **心房收缩** | 本入口 BC 在舒张末 **未见明显二次峰**；若存在则幅度很小 |

#### 重要限制（写进图注 / 汇报）

1. **未与 ECG R 波/T 波逐点对齐**——相位由 **入口流量 BC** 推断，不是心电分期。
2. **脑循环相对主动脉有延迟与阻尼**——颅内压力/WSS 峰值时刻可能略滞后于入口流量峰值。
3. **CROWN paper-original 未输入 `global_cond`（含入口流量/时相）**——模型把 81 帧当独立样本，**不显式利用** 上表相位信息；V3P/PointNetCFD 则通过 BC 特征利用时相。
4. **正式指标** 为 **81 帧 pooled**；单帧（如 1146）仅用于 **定性云图**，不单报单帧 R²。

更完整的展示帧说明见 [`outputs/field/postview/README.md`](../../../outputs/field/postview/README.md)。

**总点数 vs 每 step 采样点数**

| 口径 | 点数 | 说明 |
| --- | ---: | --- |
| 全库体素点（6561 帧） | **≈53 亿** | 6561 × ~81.5 万；数据驻留磁盘/partial pkl，训练时不整库进 GPU |
| train split 点池 | **≈37.6 亿** | 4617 帧 × ~81.5 万 |
| **每个 training step** | **最多 1.6 万（非 PINN）** | `batch_size=16` × `sample_points=10000`；每帧独立 `randperm(n)[:10000]` |
| **每个 training step（PINN）** | data 2 万 + phy 2 万坐标 | `batch_size=2`；data 与 physics **两套独立** 随机 1 万 idx |
| 正式 test 评估（paper_full） | **1,026,749,520** | 1296 帧 **全点** forward，无子采样 |

---

## 2. 训练循环（`train.py`）

### 2.1 配置常数（5751 / 5757）

| 字段 | 非 PINN 5751 | PINN 5757 |
| --- | --- | --- |
| `sample_points` | 10000 | 10000 |
| `batch_size` | **16** | **2** |
| train steps/epoch | **289** (=⌈4617/16⌉) | **2309** (=⌈4617/2⌉) |
| val steps/epoch | **41** (=⌈648/16⌉) | **324** (=⌈648/2⌉) |
| `optim.lr` | 0.003 (Adam) | 0.003 |
| `early_stopping_patience` | 50 | 50 |
| `lazy_load` / `partial_cache_cases` | true / 4 | true / 4 |

### 2.2 单 step 在做什么

1. **DataLoader** 按 `batch_size` 取出若干 **帧**（shuffle 仅 train）。
2. **`_prepare_minibatch`**（`train.py`）：对 batch 内每一帧，从 `n` 个全点中 **`torch.randperm(n)[:10000]`** 无放回随机抽点；压力标签已在 `_record_to_item` 中按 train 全集 min-max 归一化。
3. **Forward**：`CrownPointNet` — Conv1d 提局部特征 → **global max-pool** 得帧级向量 → 与每点局部特征拼接 → 回归 `u,v,w,p`。
4. **Loss & backward**：
   - **非 PINN**：`loss = MSE(pred, label)`，`autocast` + `GradScaler`。
   - **PINN**：见下节。
5. 一个 **epoch** = 遍历 train 全部 4617 帧一次；非 PINN 每 epoch 约采样 **4624 万点**（289×16×10000），相对单帧 81 万点池 **有放回重复采样**。

### 2.3 PINN 与非 PINN 的 loss 差异

| 项 | 非 PINN | PINN 5757 |
| --- | --- | --- |
| 数据项 | `loss_data = MSE(pred, label)` | 同左（用 **data idx** 采样的 1 万点） |
| 壁面项 | 无 | `loss_wall`：GT 速度 `\|v\|² ≤ 0.01` 的点上，预测 u,v,w → 0（no-slip） |
| 物理项 | 无 | `loss_phy`：在 **独立 phy idx** 的 1 万坐标上，autograd 求 NS 残差 + continuity（`Re=300`，稳态 `∂/∂t=0`） |
| 总损失 | `loss_data` | `loss = loss_data + lphy · (loss_phy + loss_wall)` |
| 坐标 | 原始 mm | `coord_scale=1000`（仅 PINN 启用时） |
| `lphy` | — | 初值 1.0；**每 10 epoch** 按 `train_loss_data / val_func` 动态更新（裁剪 [0.001, 100]） |

PINN 每 step 至少 **三次 forward**（data、wall、PDE），且 PDE 含二阶 autograd，故 5757 每 epoch 约 **1 h** 量级（5740/5757 日志），远慢于非 PINN。

---

## 3. 训练内验证 vs 正式 test 评估

### 3.1 训练内 val（`train.py` 每个 epoch 末）

- 与 train **同一套** `_prepare_minibatch`：**每帧仍随机 1 万点**（非全点）。
- `DataLoader`：`shuffle=False`，648 帧，batch 与 train 相同。
- **非 PINN**：`model.eval()` + `torch.no_grad()`，报告 `val_loss` / `val_mae`。
- **PINN**：`eval_step_pinn` 仍需 **`torch.enable_grad()`**（PDE 残差要对坐标求导），但 **无 optimizer.step**。
- 用途：**early stopping**、保存 `best_model.pt`、PINN 的 `lphy` 更新；**不是**论文正式指标。

### 3.2 正式 test（`evaluate.py` · Job 5774 / 5806）

| 项 | 训练内 val | 正式 test `eval_mode=paper_full` |
| --- | --- | --- |
| 点数/帧 | 随机 **10000** | **全点** 57万–130万 |
| Forward | batch GPU forward | 逐帧 · GPU 内 **chunk=65536** 分块 · `no_grad` |
| NMAE 分母 | 子采样 pooled range（旧 subsample 模式） | **test 全集 GT global min/max** |
| 总点数 | subsample ≈1296×10000=**1296 万** | **10.27 亿** |
| 正式结论 | ❌ 不作 Go/No-Go | ✅ **以此为准** |

训练结束 `train.py` 内也会调一次 `evaluate_checkpoint(..., eval_mode="paper")`，但汇报中的正式数字来自 **独立 Slurm evaluate**（5774/5806）。

---

## 4. 与论文原文的对齐

| 论文描述 | 本项目实现 |
| --- | --- |
| 体素 0.05 mm · 均匀化点云 | `export_pkl.py` · `voxel_size_mm=0.05` |
| 每步随机 10 000 点 | `sample_points=10000` · `_prepare_minibatch` |
| batch=1 · 15000 epoch（原文） | batch=16/2 · early stop ~105/91 epoch（AG 数据更大） |
| PINN：NS + no-slip · 动态 ω | `physics.py` + `lphy` 每 10 epoch 更新 |
| Test 全点 NMAE | `evaluate_model_paper` · `paper_full` |

---

## 5. 流程文字摘要

### 5.1 采样策略与数据层级

```
81 活跃病例（train 57 / val 8 / test 16）
  → 6561 帧样本（每病例 ≈81 个 CFD 时间步，sample_id = case/frame_key）
    → 每帧体素点云 57.5 万–130.4 万点（均值 ≈81.5 万，驻留 partial pkl / 磁盘）
      → 每个 training step：batch 内每帧 randperm(n)[:10000] 无放回随机 1 万点
        → 不整库进 GPU；全库 ≈53 亿点，正式 test 评估 10.27 亿点
```

非 PINN 5751 一个 epoch：DataLoader shuffle 遍历 train 4617 帧 · batch_size=16 → **289 steps** · 每 step 最多 16×10000=**16 万点** · epoch 合计约 **4624 万点**被采样（相对单帧点池有放回）。

### 5.2 训练主循环

```
train.py main
  → 读 config（sample_points=10000 · batch_size=16/2）
  → CrownLazyDataset（索引 jsonl · 按帧读 partial pkl）
  → DataLoader collate → 每个 epoch：
      取 batch → __getitem__ 读整帧 + train-only p min-max
      → _prepare_minibatch（每帧随机 1 万点）
      → physics.enabled?
           false（5751）：train_step_nopinn · MSE · autocast · Adam
           true（5757）：train_step_pinn · data/phy 各 1 万 idx
                         · loss_data + lphy·(loss_phy+loss_wall) · 每 10 epoch 调 lphy
      → epoch 末 val（648 帧 · 仍随机 1 万点/帧 · optimizer 不 step）
      → val_loss 更好 → best_model.pt · LR plateau · early_stop patience=50
  → 训练结束 subsample evaluate（正式指标另跑 Slurm evaluate）
```

### 5.3 验证与正式评估

**训练内 val**（每个 epoch 末，`train.py`）：

- val DataLoader：648 帧 · shuffle=False · 与 train 相同 batch_size。
- 仍走 `_prepare_minibatch`：**每帧随机 1 万点**（非全点）。
- 非 PINN：`eval_step_nopinn` · `no_grad` · 报告 val_loss/val_mae。
- PINN：`eval_step_pinn` · **`enable_grad`**（PDE 需 autograd）· 无 optimizer.step。
- 用途：early stopping、保存 best checkpoint、PINN 的 `lphy` 更新；**不作论文正式指标**。

**正式 test**（`evaluate.py` · Job 5774/5806 · `eval_mode=paper_full`）：

```
evaluate_checkpoint
  → 仅 test split（1296 帧 · lazy_load）
  → 扫描 test GT global min/max（NMAE 分母）
  → 逐帧全点 forward（57 万–130 万/帧 · GPU chunk=65536 · no_grad）
  → 汇总 10.27 亿点 · R² · NMAE · by_case/by_sample
```

口径差异：**训练 val = 子采样 1 万点/帧 + val_loss**；**正式 test = 全点 + 论文 NMAE**。汇报结论以 Slurm evaluate 的 `paper_full` 为准。

---

## 6. 关键代码位置

- 随机 1 万点采样：`train.py` → `_prepare_minibatch`
- Lazy 按帧加载：`data.py` → `CrownLazyDataset` · `build_sample_index_from_jsonl`
- 非 PINN / PINN step：`train_step_nopinn` · `train_step_pinn` · `eval_step_*`
- NS 残差：`physics.py` → `physics_residual_loss`
- 全点 test：`evaluate.py` → `evaluate_model_paper` · `_forward_full_sample`
